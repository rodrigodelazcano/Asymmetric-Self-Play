import logging
import numpy as np
import gym

from ray.rllib.utils.typing import TensorType
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, \
    normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict

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
        if permutation_invariant:
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            layers.append(nn.LayerNorm(int(out_size/2)))
        else:
            layers.append(nn.LayerNorm(out_size))

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
        
class AsymModel(TorchModelV2, nn.Module):
 
    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str,
                 **customized_model_kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.n_obj = customized_model_kwargs["number_of_objects"]

        # Policy Network
        self.rob_jt_pos_emb_pol = EmbeddingFC(obs_space["robot_joint_pos"].shape[0], 256)
        self.grip_pos_emb_pol = EmbeddingFC(obs_space["gripper_pos"].shape[0], 256)
        self.obj_state_emb_pol = nn.ModuleDict()

        for i in range(self.n_obj):
            self.obj_state_emb_pol["obj_"+str(i)+"_state"] = \
                EmbeddingFC(obs_space["obj_"+str(i)+"_state"].shape[0], 512, permutation_invariant=True)


        self.sum_pol = SumLayer(2+self.n_obj)
        self.MLP_pol = nn.Linear(256, 66)

        # Value Function Network
        self.rob_jt_pos_emb_vf = EmbeddingFC(obs_space["robot_joint_pos"].shape[0], 256)
        self.grip_pos_emb_vf = EmbeddingFC(obs_space["gripper_pos"].shape[0], 256)
        self.obj_state_emb_vf = nn.ModuleDict()

        for i in range(self.n_obj):
            self.obj_state_emb_vf["obj_"+str(i)+"_state"] = \
                EmbeddingFC(obs_space["obj_"+str(i)+"_state"].shape[0], 512, permutation_invariant=True)

        self.sum_vf = SumLayer(2+self.n_obj)
        self._value_branch = nn.Linear(256, 1)

    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        # Policy Network
        robot_joint_pos_obs = input_dict["obs"]["robot_joint_pos"].float()
        gripper_pos_obs = input_dict["obs"]["gripper_pos"].float()

        x = self.rob_jt_pos_emb_pol(robot_joint_pos_obs)
        y = self.grip_pos_emb_pol(gripper_pos_obs)
        t = torch.cat(
            [self.obj_state_emb_pol["obj_"+str(i)+"_state"](
                input_dict["obs"]["obj_"+str(i)+"_state"].float()) for i in range(self.n_obj)], -1)          

        z = torch.cat((x, y, t), -1)
        z = z.view(-1, 2+self.n_obj, 256)
        z = self.sum_pol(z)
        z = torch.squeeze(z, dim=1)
        logits = self.MLP_pol(z)

        # Value Function Network
        x_vf = self.rob_jt_pos_emb_vf(robot_joint_pos_obs)
        y_vf = self.grip_pos_emb_vf(gripper_pos_obs)
        t_vf = torch.cat(
            [self.obj_state_emb_vf["obj_"+str(i)+"_state"](
                input_dict["obs"]["obj_"+str(i)+"_state"].float()) for i in range(self.n_obj)], -1)

        z_vf = torch.cat((x_vf, y_vf, t_vf), -1)
        z_vf = z_vf.view(-1, 2+self.n_obj, 256)
        z_vf = self.sum_vf(z_vf)
        z_vf = torch.squeeze(z_vf, dim=1)
        self._value_out = self._value_branch(z_vf)
        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        return self._value_out.squeeze(1)
