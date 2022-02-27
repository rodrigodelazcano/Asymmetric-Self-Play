# from env import bob
# from env.alice import alice

# alice_env = alice.make_env()

# obs = alice_env.reset()
# # print('alice reset obs: ')

# bob_env = bob.make_env()
# init_pos = alice_env.initial_object_pos
# init_quat = alice_env.initial_object_quat

# # print('goal pos: ', init_pos)
# # print('goal quat: ', init_quat)
# # bob_env.set_initial_state_and_goal_pose(init_pos, init_quat, init_pos, init_quat)
# obs = alice_env.reset()
# episodes = 200
# while True:

#     alice_env.reset()
#     for i in range(episodes):
#         action = bob_env.action_space.sample()
#         alice_env.step(action)
#         alice_env.render()
# print('bobs reset obs: ', obs['robot_joint_pos'].shape)



# print('action space sample: ', action_space)
# print(obs['is_goal_achieved'][0])

# if not obs['is_goal_achieved'][0]:
#     print(True)

# Import the RL algorithm (Trainer) we would like to use.


from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy.sample_batch import SampleBatch
from policies.bob_policy import BobTorchPolicy
from policies.model import AsymModel
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy, policy
from ray.rllib.utils.typing import AgentID, PolicyID
from ray.rllib.models import ModelCatalog
from env.multiagent_env import AsymMultiAgent
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.tune import register_env
from multi_trajectory_collector import MultiTrajectoryCollector
from multi_episode_collector import MultiEpisodeCollector
import ray.rllib.agents.ppo as ppo
from gym import spaces
import numpy as np
import time
from typing import Dict, Optional, TYPE_CHECKING


class MyCallbacks(DefaultCallbacks):

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Optional[Dict[PolicyID, Policy]] = None, episode: Episode, **kwargs) -> None:
        # when new trajectory in agent set last_done to False (because it revives)
        alice_policy = policies['alice_policy']
        if alice_policy.new_episode_id is not None:
            alice_policy.new_episode_id = None
        for env_state in base_env.env_states:
            all_agents_done = env_state.last_dones['__all__']
            last_done_keys = env_state.last_dones.keys()
            if not all_agents_done:
                for ag in last_done_keys:
                    if ag != '__all__' and env_state.last_dones[ag]:
                        env_state.last_dones[ag] = False

number_of_objects = 2

register_env("asym_self_play",
                 lambda _: AsymMultiAgent(
                     alice_steps=100, bob_steps=200, n_objects=number_of_objects
                 ))

ModelCatalog.register_custom_model("bob_torch_model", AsymModel)

# observation_space = {}
observation_space_bob = spaces.Dict({
    "robot_joint_pos": spaces.Box(low=np.array([-6.5]*6), high=np.array([6.5]*6),dtype=np.float32),
    "gripper_pos": spaces.Box(low=np.array([-np.inf]*3), high=np.array([np.inf]*3),dtype=np.float32),
    })

for i in range(number_of_objects):
    observation_space_bob["obj_"+str(i)+"_state"] = \
        spaces.Box(low=np.array([-np.inf]*29), high=np.array([np.inf]*29), dtype=np.float32)

observation_space = spaces.Box(low=np.array([-6.5]*6), high=np.array([6.5]*6),dtype=np.float32)
action_space = spaces.MultiDiscrete(np.array([11]*6))

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    if agent_id == "alice":
        return "alice_policy"
    else:
        return "bob_policy"

def observation_fn(agent_obs, worker, base_env, policies, episode):
    return agent_obs

bob_config = {
    "model": {
        "custom_model": "bob_torch_model",
        "custom_model_config": {
            "number_of_objects": number_of_objects
        },
        # # == LSTM ==
        # # Whether to wrap the model with an LSTM.
        # "use_lstm": False,
        # # Max seq len for training the LSTM, defaults to 20.
        # "max_seq_len": 20,
        # # Size of the LSTM cell.
        # "lstm_cell_size": 256,
        # # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
        # "lstm_use_prev_action": False,
        # # Whether to feed r_{t-1} to LSTM.
        # "lstm_use_prev_reward": False,
        # # Whether the LSTM is time-major (TxBx..) or batch-major (BxTx..).
        # "_time_major": False,
    }
}

config = {
    "num_workers": 1,
    "num_envs_per_worker": 3,
    "rollout_fragment_length": 5000,
    "batch_mode": "complete_episodes",
    "framework": "torch",
    "train_batch_size": 40000,
    "sgd_minibatch_size": 50,
    "multiagent": {
        "policies": {
            "alice_policy": (PPOTorchPolicy, observation_space, action_space, {}),
            "bob_policy": policy.PolicySpec(policy_class=BobTorchPolicy, 
                                                observation_space = observation_space_bob, 
                                                action_space = action_space, 
                                                config=bob_config
                                            ),
            # "bob_polic y": (PPOTorchPolicy, observation_space, action_space, {}),
        },
        "policy_mapping_fn": policy_mapping_fn,
        "observation_fn": observation_fn,
    },
    # "callbacks": MyCallbacks,
    "sample_collector": MultiEpisodeCollector,
    "_disable_preprocessor_api": True,
}   

# Create our RLlib Trainer.
trainer = ppo.PPOTrainer(env="asym_self_play", config=config)


for _ in range(3):
    results = trainer.train()
    print(results)

