import logging
import numpy as np
import gym

from ray.rllib.utils.typing import TensorType
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import one_hot
from typing import Dict, List, Union
from ray.rllib.utils.typing import TensorType, ModelConfigDict

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)

class EmbeddingFC(nn.Module):
    
    def __init__(self,
                 in_size: int,
                 out_size: int,
                 use_bias: bool = False,
                 bias_init: float  = 0.0,
                 permutation_invariant: bool = False):
        super(EmbeddingFC, self).__init__()

        layers = []
        layers.append(nn.Linear(in_size, out_size, bias=use_bias))
        # if permutation_invariant:
        #     layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
        #     layers.append(nn.LayerNorm(int(out_size/2)))
        # else:
        #     layers.append(nn.LayerNorm(out_size))

        self._model = nn.Sequential(*layers)
    
    def forward(self, x: TensorType) -> TensorType:
 
        return self._model(x)

class SumLayer(nn.Module):
    def __init__(self,
                 in_size: int):
        super(SumLayer, self).__init__()

        layers = []
        layers.append(nn.Conv1d(in_channels=in_size, out_channels=1, kernel_size=1))
        layers.append(nn.ReLU())

        self._model = nn.Sequential(*layers)

    def forward(self, x: TensorType) -> TensorType:
        return self._model(x)

class MLP(nn.Module):
    def __init__(self,
                hiddens: List[int],
                activation_func: str,
                prev_layer_size: int):
        super(MLP, self).__init__()
        layers = []

        for size in hiddens:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation_func
                )
            )
            prev_layer_size = size
        
        self._model = nn.Sequential(*layers)


    def forward(self, x: TensorType) -> TensorType:
        return self._model(x)
class AsymModel(TorchRNN, nn.Module):
 
    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str,
                 **customized_model_kwargs):
        nn.Module.__init__(self) 
        super().__init__(obs_space, action_space, num_outputs,
                              model_config, name)
        
        self.n_obj = customized_model_kwargs["number_of_objects"]
        self.cell_size = model_config["lstm_cell_size"]
        self.time_major = model_config.get("_time_major", False)
        self.use_prev_action = model_config["lstm_use_prev_action"]
        self.use_prev_reward = model_config["lstm_use_prev_reward"]

        self.mlp_hiddens = customized_model_kwargs["mlp_hiddens"]
        self.mlp_activation = customized_model_kwargs["mlp_activation"]

        self.embedding_size = 256

        self.dict_obs_space = customized_model_kwargs["dict_obs_space"]

        # Add prev-action/reward nodes to input to LSTM.
        self.mlp_num_outputs = self.mlp_hiddens[-1]
        if self.use_prev_action:
            self.action_dim = np.sum(action_space.nvec)
            self.mlp_num_outputs += self.action_dim
        if self.use_prev_reward:
            self.mlp_num_outputs += 1

        # Define LSTM for both, PolicyNetwork and ValueFunction
        self.lstm_pol = nn.LSTM(
            self.mlp_num_outputs, self.cell_size, batch_first=not self.time_major
            )
        self.lstm_val = nn.LSTM(
            self.mlp_num_outputs, self.cell_size, batch_first=not self.time_major
            )
        
        ## === Policy Network === 
        # Embeddings
        self.rob_jt_pos_emb_pol = EmbeddingFC(self.dict_obs_space["robot_joint_pos"].shape[0], self.embedding_size)
        self.grip_pos_emb_pol = EmbeddingFC(self.dict_obs_space["gripper_pos"].shape[0], self.embedding_size)
        self.obj_state_emb_pol = nn.ModuleDict()

        # Layer Norms
        self.robot_jt_pos_ly_norm_pol = nn.LayerNorm(self.embedding_size)
        self.grip_pos_ly_norm_pol = nn.LayerNorm(self.embedding_size)
        self.obj_state_ly_norm_pol = nn.ModuleDict()

        # MaxPool
        self.maxpool_pol = nn.ModuleDict()
        
        for i in range(self.n_obj):
            self.obj_state_emb_pol["obj_"+str(i)+"_state"] = \
                EmbeddingFC(self.dict_obs_space["obj_"+str(i)+"_state"].shape[0], self.embedding_size*2, permutation_invariant=True)
            self.obj_state_ly_norm_pol["obj_"+str(i)+"_state"] = nn.LayerNorm(self.embedding_size)
            self.maxpool_pol["obj_"+str(i)+"_state"] = nn.MaxPool1d(kernel_size=2, stride=2)          
        
        self.sum_pol = SumLayer(in_size=2+self.n_obj)
        self.mlp_pol = MLP(self.mlp_hiddens,
                            self.mlp_activation,
                            self.embedding_size)
        self._logits_branch = SlimFC(
            in_size=self.cell_size,
            out_size=66,
            activation_fn=None,
            initializer=nn.init.xavier_uniform_)

        ## === Value Function Network ===
        # Embeddings
        self.rob_jt_pos_emb_vf = EmbeddingFC(self.dict_obs_space["robot_joint_pos"].shape[0], self.embedding_size)
        self.grip_pos_emb_vf = EmbeddingFC(self.dict_obs_space["gripper_pos"].shape[0], self.embedding_size)
        self.obj_state_emb_vf = nn.ModuleDict()

        # Layer Norms
        self.robot_jt_pos_ly_norm_vf = nn.LayerNorm(self.embedding_size)
        self.grip_pos_ly_norm_vf = nn.LayerNorm(self.embedding_size)
        self.obj_state_ly_norm_vf = nn.ModuleDict()

        # MaxPool
        self.maxpool_vf = nn.ModuleDict()

        for i in range(self.n_obj):
            self.obj_state_emb_vf["obj_"+str(i)+"_state"] = \
                EmbeddingFC(self.dict_obs_space["obj_"+str(i)+"_state"].shape[0], self.embedding_size*2, permutation_invariant=True)
            self.obj_state_ly_norm_vf["obj_"+str(i)+"_state"] = nn.LayerNorm(self.embedding_size)
            self.maxpool_vf["obj_"+str(i)+"_state"] = nn.MaxPool1d(kernel_size=2, stride=2)

        self.sum_vf = SumLayer(in_size = 2 + self.n_obj)
        self.mlp_vf = MLP(self.mlp_hiddens,
                            self.mlp_activation,
                            self.embedding_size)
        self._value_branch = SlimFC(
            in_size=self.cell_size,
            out_size=1,
            activation_fn=None,
            initializer=nn.init.xavier_uniform_)

        if model_config["lstm_use_prev_action"]:
            self.view_requirements[SampleBatch.PREV_ACTIONS] = \
                ViewRequirement(SampleBatch.ACTIONS, space=self.action_space,
                                shift=-1)
        if model_config["lstm_use_prev_reward"]:
            self.view_requirements[SampleBatch.PREV_REWARDS] = \
                ViewRequirement(SampleBatch.REWARDS, shift=-1)

    @override(TorchRNN)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        assert seq_lens is not None
        # Concat. prev-action/reward if required.
        prev_a_r = []
        if self.model_config["lstm_use_prev_action"]:
            prev_a = one_hot(input_dict[SampleBatch.PREV_ACTIONS].float(),
                                 self.action_space)
            prev_a_r.append(torch.reshape(prev_a, [-1, self.action_dim]))

        if self.model_config["lstm_use_prev_reward"]:
            prev_a_r.append(
                torch.reshape(input_dict[SampleBatch.PREV_REWARDS].float(),
                              [-1, 1]))

        ## === Policy Network ===
        robot_joint_pos_obs = input_dict["obs"]["robot_joint_pos"].float()
        gripper_pos_obs = input_dict["obs"]["gripper_pos"].float()

        x = self.rob_jt_pos_emb_pol(robot_joint_pos_obs)
        y = self.grip_pos_emb_pol(gripper_pos_obs)

        x = self.robot_jt_pos_ly_norm_pol(x)
        y = self.grip_pos_ly_norm_pol(y)

        obj_state_emb_ouput = []
        for i in range(self.n_obj):
            emb_out = self.obj_state_emb_pol["obj_"+str(i)+"_state"](
                input_dict["obs"]["obj_"+str(i)+"_state"].float())
            emb_out = emb_out.unsqueeze(1)
            emb_out = self.maxpool_pol["obj_"+str(i)+"_state"](emb_out)
            emb_out = emb_out.squeeze(1)
            obj_state_emb_ouput.append(self.obj_state_ly_norm_pol["obj_"+str(i)+"_state"](emb_out))
        t = torch.cat(obj_state_emb_ouput, -1)          

        z = torch.cat((x, y, t), -1)
        z = z.view(-1, 2+self.n_obj, self.embedding_size)
        z = self.sum_pol(z)
        z = torch.squeeze(z, dim=1)
        z = self.mlp_pol(z)
        if prev_a_r:
            z = torch.cat([z] + prev_a_r, dim=1)

        inputs_lstm_pol = self._add_time_dimension_to_batch(z, seq_lens)
        self._features, [h_pol, c_pol] = self.lstm_pol(
            inputs_lstm_pol, [torch.unsqueeze(state[0], 0),
                              torch.unsqueeze(state[1], 0)]
        )
        logits = self._logits_branch(self._features)

        ## === Value Function Network ===
        x_vf = self.rob_jt_pos_emb_vf(robot_joint_pos_obs)
        y_vf = self.grip_pos_emb_vf(gripper_pos_obs)
        x_vf = self.robot_jt_pos_ly_norm_pol(x_vf)
        y_vf = self.grip_pos_ly_norm_pol(y_vf)

        obj_state_emb_ouput = []
        for i in range(self.n_obj):
            emb_out = self.obj_state_emb_vf["obj_"+str(i)+"_state"](
                input_dict["obs"]["obj_"+str(i)+"_state"].float())
            emb_out = emb_out.unsqueeze(1)
            emb_out = self.maxpool_vf["obj_"+str(i)+"_state"](emb_out)
            emb_out = emb_out.squeeze(1)
            obj_state_emb_ouput.append(self.obj_state_ly_norm_vf["obj_"+str(i)+"_state"](emb_out))
            
        t_vf = torch.cat(obj_state_emb_ouput, -1) 

        z_vf = torch.cat((x_vf, y_vf, t_vf), -1)
        z_vf = z_vf.view(-1, 2 + self.n_obj, self.embedding_size)
        z_vf = self.sum_vf(z_vf)
        z_vf = torch.squeeze(z_vf, dim=1)

        z_vf = self.mlp_vf(z_vf)

        if prev_a_r:
            z_vf = torch.cat([z_vf] + prev_a_r, dim=1)
        inputs_lstm_val = self._add_time_dimension_to_batch(z_vf, seq_lens)
        self._value_out, [h_vf, c_vf] = self.lstm_val(
            inputs_lstm_val, [torch.unsqueeze(state[2], 0),
                              torch.unsqueeze(state[3], 0)]
        )
        output = torch.reshape(logits, [-1, 66])
        return output, [torch.squeeze(h_pol, 0), torch.squeeze(c_pol, 0), torch.squeeze(h_vf, 0), torch.squeeze(c_vf, 0)]

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._value_out is not None, "must call forward() first"
        return torch.reshape(self._value_branch(self._value_out), [-1]) 
    
    @override(ModelV2)
    def get_initial_state(self) -> Union[List[np.ndarray], List[TensorType]]:
        # Place hidden states on same device as model.
        linear_logits = next(self._logits_branch._model.children())
        linear_values = next(self._value_branch._model.children())
        h = [
            linear_logits.weight.new(1, self.cell_size).zero_().squeeze(0),
            linear_logits.weight.new(1, self.cell_size).zero_().squeeze(0),
            linear_values.weight.new(1, self.cell_size).zero_().squeeze(0),
            linear_values.weight.new(1, self.cell_size).zero_().squeeze(0)
        ]
        return h

    def _add_time_dimension_to_batch(self, input: TensorType,
                                     seq_lens: TensorType) -> TensorType:
        
        """Adds time dimension to batch before sending inputs to forward_rnn()."""

        flat_inputs = input.float()
        if isinstance(seq_lens, np.ndarray):
            seq_lens = torch.Tensor(seq_lens).int()
        max_seq_len = flat_inputs.shape[0] // seq_lens.shape[0]
        self.time_major = self.model_config.get("_time_major", False)
        inputs = add_time_dimension(
            flat_inputs,
            max_seq_len=max_seq_len,
            framework="torch",
            time_major=self.time_major,
        )
        return inputs
