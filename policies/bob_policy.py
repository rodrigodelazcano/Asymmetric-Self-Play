from email import policy
import logging
from typing import Dict, List, Type, Union
import numpy as np
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy 
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import explained_variance, sequence_mask
from ray.rllib.utils.typing import TensorType
from ray.rllib.policy.rnn_sequencing import timeslice_along_seq_lens_with_overlap
from ray.rllib.models.modelv2 import restore_original_dimensions
from robogym.envs.rearrange.goals.object_state import full_euler_angle_difference
from robogym.utils import rotation
torch, nn = try_import_torch()

logger = logging.getLogger(__name__)


class BobTorchPolicy(PPOTorchPolicy):
    """PyTorch policy class used with PPOTrainer for Bob's policy."""

    @override(PPOTorchPolicy)
    def __init__(self, observation_space, action_space, config):
        # Constant behavioral clonning
        self.beta = config["ABC_loss_weight"]
        # Observation space shape
        self.obs_shape = observation_space.shape[0]
        super().__init__(observation_space, action_space, config)        

    @override(PPOTorchPolicy)
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

            bc_mask, bc_num_valid = self.loss_mask(train_batch[SampleBatch.AGENT_INDEX],
                                    mask,
                                    max_seq_len,
                                    time_major=model.is_time_major(),
                                    loss="bc")
            ppo_mask, ppo_num_valid = self.loss_mask(train_batch[SampleBatch.AGENT_INDEX],
                                    mask,
                                    max_seq_len,
                                    time_major=model.is_time_major(),
                                    loss="ppo")
            
            def reduce_mean_valid_multi_loss(t, loss):
                if loss == "bc":
                    if bc_num_valid != 0:
                        return torch.sum(t[bc_mask]) / bc_num_valid
                    else:
                        return torch.tensor([0.0], requires_grad=False, device=self.device)
                else:
                    if ppo_num_valid != 0:
                        return torch.sum(t[ppo_mask]) / ppo_num_valid
                    else:
                        return torch.tensor([0.0],  requires_grad=False, device=self.device)
                

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

        mean_kl_loss = reduce_mean_valid_multi_loss(action_kl, loss="ppo")

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = reduce_mean_valid_multi_loss(curr_entropy, loss="ppo")

        surrogate_loss = torch.min(
            train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
            train_batch[Postprocessing.ADVANTAGES] * torch.clamp(
                logp_ratio, 1 - self.config["clip_param"],
                1 + self.config["clip_param"]))

        mean_ppo_policy_loss = reduce_mean_valid_multi_loss(-surrogate_loss, loss="ppo")
        mean_bc_policy_loss = reduce_mean_valid_multi_loss(-surrogate_loss, loss="bc")

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
            mean_vf_loss = reduce_mean_valid_multi_loss(vf_loss, loss="ppo")
        # Ignore the value function.
        else:
            vf_loss = mean_vf_loss = 0.0

        policy_loss = reduce_mean_valid_multi_loss(-surrogate_loss +
                                       self.kl_coeff * action_kl +
                                       self.config["vf_loss_coeff"] * vf_loss -
                                       self.entropy_coeff * curr_entropy, loss="ppo")
        bc_loss = self.beta*reduce_mean_valid_multi_loss(-surrogate_loss, loss="bc") 

        total_loss =  policy_loss + bc_loss
                        
        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_ppo_policy_loss"] = mean_ppo_policy_loss
        model.tower_stats["mean_bc_policy_loss"] = mean_bc_policy_loss
        model.tower_stats["mean_vf_loss"] = mean_vf_loss
        model.tower_stats["vf_explained_var"] = explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], model.value_function())
        model.tower_stats["mean_entropy"] = mean_entropy
        model.tower_stats["mean_kl_loss"] = mean_kl_loss

        return total_loss

    @override(PPOTorchPolicy)
    def extra_grad_info(self,
                        train_batch: SampleBatch) -> Dict[str, TensorType]:
        return convert_to_numpy({
            "cur_kl_coeff": self.kl_coeff,
            "cur_lr": self.cur_lr,
            "total_loss": torch.mean(
                torch.stack(self.get_tower_stats("total_loss"))),
            "ppo_policy_loss": torch.mean(
                torch.stack(self.get_tower_stats("mean_ppo_policy_loss"))),
            "bc_policy_loss": torch.mean(
                torch.stack(self.get_tower_stats("mean_bc_policy_loss"))),
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

    def relable_demonstration(self, demonstration_batch, alice_observation_space):
        
        with torch.no_grad():
            if self.model:
                self.model.eval()
            
            batch_size = demonstration_batch.count
            obs_demonstration_batch = demonstration_batch[SampleBatch.OBS].copy()

            original_dict_obs = restore_original_dimensions(obs_demonstration_batch, alice_observation_space, tensorlib="numpy")

            # Add current and relative goal state to observations
            last_obj_pos = demonstration_batch['infos'][-1]['last_object_pos'].copy()
            last_obj_rot = demonstration_batch['infos'][-1]['last_object_eul'].copy()
            for i in range(last_obj_pos.shape[0]):
                relative_goal_state = self._get_relative_goal_distance(last_obj_pos[i], 
                                                                        last_obj_rot[i], 
                                                                        original_dict_obs["obj_"+str(i)+ "_state"][:,0:3], 
                                                                        original_dict_obs["obj_"+str(i)+ "_state"][:,3:6], 
                                                                        batch_size)

                goal_state = np.concatenate((np.tile(last_obj_pos[i], (batch_size, 1)), 
                                                np.tile(last_obj_rot[i], (batch_size, 1)),
                                                relative_goal_state["obj_pos"], 
                                                relative_goal_state["obj_rot"]),
                                                axis=1)
                obj_state = original_dict_obs["obj_" + str(i) + "_state"].copy()

                new_obj_state_obs = np.concatenate((obj_state, goal_state), axis=1)
                original_dict_obs["obj_" + str(i) + "_state"] = new_obj_state_obs

            flatten_bob_obs = self.flatten_obs(original_dict_obs, batch_size)

            demonstration_batch[SampleBatch.OBS] = flatten_bob_obs

            # Add current and relative goal state to new observations key
            new_obs = flatten_bob_obs[1:].copy()
            new_obs_demonstration_batch = demonstration_batch[SampleBatch.NEXT_OBS][-1].copy()
            original_dict_new_obs = restore_original_dimensions(np.reshape(new_obs_demonstration_batch, (1, new_obs_demonstration_batch.shape[0])),
                                                                    alice_observation_space, 
                                                                    tensorlib="numpy")

            for i in range(last_obj_pos.shape[0]):
                relative_goal_state = self._get_relative_goal_distance(last_obj_pos[i], 
                                                                        last_obj_rot[i], 
                                                                        np.reshape(original_dict_new_obs["obj_"+str(i)+ "_state"][-1,0:3],(1, 3)), 
                                                                        np.reshape(original_dict_new_obs["obj_"+str(i)+ "_state"][-1,3:6], (1, 3)),
                                                                        batch_size=1)

                goal_state = np.concatenate((np.reshape(last_obj_pos[i], (1, 3)), 
                                                np.reshape(last_obj_rot[i], (1, 3)),
                                                relative_goal_state["obj_pos"], 
                                                relative_goal_state["obj_rot"]),
                                                axis=1)
                obj_state = np.reshape(original_dict_new_obs["obj_" + str(i) + "_state"][-1].copy(), (1, 17))

                new_obj_state_new_obs = np.concatenate((obj_state, goal_state), axis=1)
                original_dict_new_obs["obj_" + str(i) + "_state"] = new_obj_state_new_obs

            last_new_obs = self.flatten_obs(original_dict_new_obs, 1)
            
            flatten_bob_new_obs = np.concatenate((new_obs, last_new_obs), axis=0)
            demonstration_batch[SampleBatch.NEXT_OBS] = flatten_bob_new_obs
            demonstration_batch[Postprocessing.ADVANTAGES] = np.ones((batch_size, ), dtype=int)

            # Get Bob's action distributions for each new observation
            if self._is_recurrent:
                input_batch_list = timeslice_along_seq_lens_with_overlap(demonstration_batch)
            else:
                input_batch_list = [demonstration_batch]

            state_out = None
            for i, batch in enumerate(input_batch_list):
                if (SampleBatch.PREV_REWARDS in batch.keys()) and (not self.config["model"]["lstm_use_prev_reward"]):
                    del batch[SampleBatch.PREV_REWARDS]
                input_batch = self._lazy_tensor_dict(batch)
                if i != 0:
                    for j, state in enumerate(state_out):
                        input_batch["state_in_" + str(j)] = state

                dist_class = self.dist_class
                logits, state_out = self.model(input_batch)

                for j, state in enumerate(state_out):
                    input_batch["state_out_" + str(j)] = state           
                action_dist = dist_class(logits, self.model)
                input_batch[SampleBatch.ACTION_LOGP] = action_dist.logp(input_batch[SampleBatch.ACTIONS])
                input_batch[SampleBatch.AGENT_INDEX] = torch.add(input_batch[SampleBatch.AGENT_INDEX], 2)

        return SampleBatch.concat_samples(input_batch_list)

    def _get_relative_goal_distance(self, goal_pos, goal_rot, current_obj_pos, current_obj_rot, batch_size):
        
        goal_state = {}
        goal_state["obj_pos"] = np.tile(goal_pos, (batch_size, 1))
        goal_state["obj_rot"] = np.tile(goal_rot, (batch_size, 1))

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

    def loss_mask(self, 
        agent_idx,
        pad_mask,
        dtype=None,
        time_major = False,
        loss = "bc"
        ):

        # Calculate loss mask for behavioral cloning samples
        if loss == "bc":
            mask = agent_idx == 2
        # Calculate loss mask for PPO samples
        else:
            mask = agent_idx != 2

        if not time_major:
            mask = mask.t()

        mask *= pad_mask

        num_valid = torch.sum(mask)

        return mask, num_valid
