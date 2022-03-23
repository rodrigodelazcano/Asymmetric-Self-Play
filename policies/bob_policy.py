import enum
import logging
from typing import Dict, List, Type, Union
from matplotlib.pyplot import axis
import numpy as np
import ray
from ray.rllib.agents.ppo.ppo_tf_policy import setup_config
from ray.rllib.evaluation.postprocessing import compute_gae_for_sample_batch, \
    Postprocessing
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy import EntropyCoeffSchedule, \
    LearningRateSchedule, TorchPolicy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import apply_grad_clipping, \
    explained_variance, sequence_mask
from ray.rllib.utils.typing import TensorType
from ray.rllib.policy.rnn_sequencing import timeslice_along_seq_lens_with_overlap
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.preprocessors import DictFlatteningPreprocessor
from robogym.envs.rearrange.goals.object_state import full_euler_angle_difference
from robogym.utils import rotation
torch, nn = try_import_torch()

logger = logging.getLogger(__name__)


class BobTorchPolicy(TorchPolicy, LearningRateSchedule, EntropyCoeffSchedule):
    """PyTorch policy class used with PPOTrainer."""

    def __init__(self, observation_space, action_space, config):
        config = dict(ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG, **config)
        setup_config(self, observation_space, action_space, config)

        TorchPolicy.__init__(
            self,
            observation_space,
            action_space,
            config,
            max_seq_len=config["model"]["max_seq_len"])

        EntropyCoeffSchedule.__init__(self, config["entropy_coeff"],
                                      config["entropy_coeff_schedule"])
        LearningRateSchedule.__init__(self, config["lr"],
                                      config["lr_schedule"])

        # The current KL value (as python float).
        self.kl_coeff = self.config["kl_coeff"]
        # Constant target value.
        self.kl_target = self.config["kl_target"]\

        self.obs_shape = observation_space.shape[0]

        # dict_obs_space = config["model"]["custom_model_config"]["dict_obs_space"]
        # self.preprocessor = DictFlatteningPreprocessor(dict_obs_space)

        # TODO: Don't require users to call this manually.
        self._initialize_loss_from_dummy_batch()
        

    @override(TorchPolicy)
    def postprocess_trajectory(self,
                               sample_batch,
                               other_agent_batches=None,
                               episode=None):
        with torch.no_grad():
            return compute_gae_for_sample_batch(self, sample_batch,
                                                other_agent_batches, episode)

    # TODO: Add method to Policy base class (as the new way of defining loss
    #  functions (instead of passing 'loss` to the super's constructor)).
    @override(TorchPolicy)
    def loss(self, model: ModelV2, dist_class: Type[ActionDistribution],
             train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
        """Constructs the loss for Proximal Policy Objective.

        Args:
            model: The Model to calculate the loss for.
            dist_class: The action distr. class.
            train_batch: The training data.

        Returns:
            The PPO loss tensor given the input batch.
        """

        logits, state = model(train_batch)
        curr_action_dist = dist_class(logits, model)


        # RNN case: Mask away 0-padded chunks at end of time axis.
        if state:
            B = len(train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = logits.shape[0] // B
            mask = sequence_mask(
                train_batch[SampleBatch.SEQ_LENS],
                max_seq_len,
                time_major=model.is_time_major())
            mask = torch.reshape(mask, [-1])
            num_valid = torch.sum(mask)

            def reduce_mean_valid(t):
                return torch.sum(t[mask]) / num_valid

        # non-RNN case: No masking.
        else:
            mask = None
            reduce_mean_valid = torch.mean

        prev_action_dist = dist_class(
            train_batch[SampleBatch.ACTION_DIST_INPUTS], model)

        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS]) -
            train_batch[SampleBatch.ACTION_LOGP])
        action_kl = prev_action_dist.kl(curr_action_dist)
        mean_kl_loss = reduce_mean_valid(action_kl)

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = reduce_mean_valid(curr_entropy)

        surrogate_loss = torch.min(
            train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
            train_batch[Postprocessing.ADVANTAGES] * torch.clamp(
                logp_ratio, 1 - self.config["clip_param"],
                1 + self.config["clip_param"]))
        # print('SURROGATE LOSS: ', surrogate_loss)
        mean_policy_loss = reduce_mean_valid(-surrogate_loss)
        # print('MEAN POLICY LOSS: ', mean_policy_loss)
        # Compute a value function loss.
        if self.config["use_critic"]:
            prev_value_fn_out = train_batch[SampleBatch.VF_PREDS]
            value_fn_out = model.value_function()
            vf_loss1 = torch.pow(
                value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
            vf_clipped = prev_value_fn_out + torch.clamp(
                value_fn_out - prev_value_fn_out,
                -self.config["vf_clip_param"], self.config["vf_clip_param"])
            vf_loss2 = torch.pow(
                vf_clipped - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
            vf_loss = torch.max(vf_loss1, vf_loss2)
            mean_vf_loss = reduce_mean_valid(vf_loss)
        # Ignore the value function.
        else:
            vf_loss = mean_vf_loss = 0.0

        total_loss = reduce_mean_valid(-surrogate_loss +
                                       self.kl_coeff * action_kl +
                                       self.config["vf_loss_coeff"] * vf_loss -
                                       self.entropy_coeff * curr_entropy)

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_policy_loss"] = mean_policy_loss
        model.tower_stats["mean_vf_loss"] = mean_vf_loss
        model.tower_stats["vf_explained_var"] = explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], model.value_function())
        model.tower_stats["mean_entropy"] = mean_entropy
        model.tower_stats["mean_kl_loss"] = mean_kl_loss

        return total_loss

    def _value(self, **input_dict):
        # When doing GAE, we need the value function estimate on the
        # observation.
        if self.config["use_gae"]:
            # Input dict is provided to us automatically via the Model's
            # requirements. It's a single-timestep (last one in trajectory)
            # input_dict.
            input_dict = self._lazy_tensor_dict(input_dict)
            model_out, _ = self.model(input_dict)
            # [0] = remove the batch dim.
            return self.model.value_function()[0].item()
        # When not doing GAE, we do not require the value function's output.
        else:
            return 0.0

    def update_kl(self, sampled_kl):
        # Update the current KL value based on the recently measured value.
        if sampled_kl > 2.0 * self.kl_target:
            self.kl_coeff *= 1.5
        elif sampled_kl < 0.5 * self.kl_target:
            self.kl_coeff *= 0.5
        # Return the current KL value.
        return self.kl_coeff

    # TODO: Make this an event-style subscription (e.g.:
    #  "after_actions_computed").
    @override(TorchPolicy)
    def extra_action_out(self, input_dict, state_batches, model, action_dist):
        # Return value function outputs. VF estimates will hence be added to
        # the SampleBatches produced by the sampler(s) to generate the train
        # batches going into the loss function.
        return {
            SampleBatch.VF_PREDS: model.value_function(),
        }

    # TODO: Make this an event-style subscription (e.g.:
    #  "after_gradients_computed").
    @override(TorchPolicy)
    def extra_grad_process(self, local_optimizer, loss):
        return apply_grad_clipping(self, local_optimizer, loss)

    # TODO: Make this an event-style subscription (e.g.:
    #  "after_losses_computed").
    @override(TorchPolicy)
    def extra_grad_info(self,
                        train_batch: SampleBatch) -> Dict[str, TensorType]:
        return convert_to_numpy({
            "cur_kl_coeff": self.kl_coeff,
            "cur_lr": self.cur_lr,
            "total_loss": torch.mean(
                torch.stack(self.get_tower_stats("total_loss"))),
            "policy_loss": torch.mean(
                torch.stack(self.get_tower_stats("mean_policy_loss"))),
            "vf_loss": torch.mean(
                torch.stack(self.get_tower_stats("mean_vf_loss"))),
            "vf_explained_var": torch.mean(
                torch.stack(self.get_tower_stats("vf_explained_var"))),
            "kl": torch.mean(
                torch.stack(self.get_tower_stats("mean_kl_loss"))),
            "entropy": torch.mean(
                torch.stack(self.get_tower_stats("mean_entropy"))),
            "entropy_coeff": self.entropy_coeff,
        })

    # TODO: Make lr-schedule and entropy-schedule Plugin-style functionalities
    #  that can be added (via the config) to any Trainer/Policy.
    @override(TorchPolicy)
    def on_global_var_update(self, global_vars):
        super().on_global_var_update(global_vars)
        if self._lr_schedule:
            self.cur_lr = self._lr_schedule.value(global_vars["timestep"])
            for opt in self._optimizers:
                for p in opt.param_groups:
                    p["lr"] = self.cur_lr
        if self._entropy_coeff_schedule is not None:
            self.entropy_coeff = self._entropy_coeff_schedule.value(
                global_vars["timestep"])
    
    def relable_demonstration(self, demonstration_batch, alice_observation_space):
        
        with torch.no_grad():
            if self.model:
                self.model.eval()
            
            batch_size = demonstration_batch.count
            obs_demonstration_batch = demonstration_batch["obs"].copy()

            original_dict_obs = restore_original_dimensions(obs_demonstration_batch, alice_observation_space, tensorlib="numpy")

            # Add current and relative goal state to observations
            last_obj_pos = demonstration_batch['infos'][-1]['last_object_pos'].copy()
            last_obj_rot = demonstration_batch['infos'][-1]['last_object_eul'].copy()
            for i in range(last_obj_pos.shape[0]):
                relative_goal_state = self._get_relative_goal_distance(last_obj_pos[i], last_obj_rot[i], 
                                                                                original_dict_obs["obj_"+str(i)+ "_state"][:,0:3], original_dict_obs["obj_"+str(i)+ "_state"][:,3:6])

                goal_state = np.concatenate((np.tile(last_obj_pos[i], (batch_size, 1)), 
                                                np.tile(last_obj_rot[i], (batch_size, 1)),
                                                relative_goal_state["obj_pos"], relative_goal_state["obj_rot"]),
                                                axis=1)
                obj_state = original_dict_obs["obj_" + str(i) + "_state"].copy()

                new_obj_state_obs = np.concatenate((obj_state, goal_state), axis=1)
                original_dict_obs["obj_" + str(i) + "_state"] = new_obj_state_obs

            flatten_bob_obs = self.flatten_obs(original_dict_obs, batch_size)

            demonstration_batch["obs"] = flatten_bob_obs
                  
            # Get Bob's action distributions for each new observation
            if self._is_recurrent:
                input_batch_list = timeslice_along_seq_lens_with_overlap(demonstration_batch)
            else:
                input_batch_list = [demonstration_batch]

            state_out = None
            for i, batch in enumerate(input_batch_list):
                input_batch = self._lazy_tensor_dict(batch)
                if i != 0:
                    for j, state in enumerate(state_out):
                        input_batch["state_in_" + str(j)] = state

                dist_class = self.dist_class
                dist_inputs, state_out = self.model(input_batch)

                for j, state in enumerate(state_out):
                    input_batch["state_out_" + str(j)] = state
            breakpoint()


    def _get_relative_goal_distance(self, goal_pos, goal_rot, current_obj_pos, current_obj_rot):
        
        goal_state = {}
        goal_state["obj_pos"] = np.tile(goal_pos, (current_obj_pos.shape[0], 1))
        goal_state["obj_rot"] = np.tile(goal_rot, (current_obj_rot.shape[0], 1))

        current_state = {}
        current_state["obj_pos"] = current_obj_pos
        current_state["obj_rot"] = current_obj_rot

        # All the objects are different.
        relative_obj_pos = goal_state["obj_pos"] - current_state["obj_pos"]
        relative_obj_rot = full_euler_angle_difference(goal_state, current_state)
        # normalize angles
        relative_obj_rot = rotation.normalize_angles(relative_obj_rot)

        return {
            "obj_pos": relative_obj_pos.copy(),
            "obj_rot": relative_obj_rot.copy(),
        }

    def flatten_obs(self, obs_dict, batch_size):
        flatten_obs = np.zeros((batch_size, self.obs_shape), dtype=np.float32)
        offset = 0
        for key, obs in obs_dict.items():
            flatten_obs[:, offset:offset + obs.shape[1]] = obs
            offset += obs.shape[1] 
        return flatten_obs    
