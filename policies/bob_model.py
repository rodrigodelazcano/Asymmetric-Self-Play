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
                 bias_init: float  = 0.0):
        super(EmbeddingFC, self).__init__()

        layers = []
        layers.append(nn.Linear(in_size, out_size, bias=use_bias))
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
        
class BobModel(TorchModelV2, nn.Module):
 
    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        # Policy Network
        self.rob_jt_pos_emb_pol = EmbeddingFC(obs_space.original_space["robot_joint_pos"].shape[0], 256)
        self.grip_pos_emb_pol = EmbeddingFC(obs_space.original_space["gripper_pos"].shape[0], 256)
        self.sum_pol = SumLayer(2)
        self.MLP_pol = nn.Linear(256, 66)

        # Value Function Network
        self.rob_jt_pos_emb_vf = EmbeddingFC(obs_space.original_space["robot_joint_pos"].shape[0], 256)
        self.grip_pos_emb_vf = EmbeddingFC(obs_space.original_space["gripper_pos"].shape[0], 256)
        self.sum_vf = SumLayer(2)
        self.MLP_vf = nn.Linear(256, 1)

    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):

        self.robot_joint_pos_obs = input_dict["obs"]["robot_joint_pos"].float()
        self.gripper_pos_obs = input_dict["obs"]["gripper_pos"].float()

        x = self.rob_jt_pos_emb_pol(self.robot_joint_pos_obs)
        y = self.grip_pos_emb_pol(self.gripper_pos_obs)

        z = torch.cat((x, y), -1)
        z = z.view(-1, 2, 256)
        z = self.sum_pol(z)
        z = torch.squeeze(z)
        logits = self.MLP_pol(z)

        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:

        x = self.rob_jt_pos_emb_vf(self.robot_joint_pos_obs)
        y = self.grip_pos_emb_vf(self.gripper_pos_obs)

        z = torch.cat((x, y), -1)
        z = z.view(-1, 2, 256)
        z = self.sum_vf(z)
        z = torch.squeeze(z)

        return self.MLP_vf(z).squeeze(1)
