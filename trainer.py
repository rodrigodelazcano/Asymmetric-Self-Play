# from env import bob
# from env.alice import alice
import colorama
from colorama import Fore

# alice_env = alice.make_env()

# obs = alice_env.reset()
# print('alice reset obs: ')

# bob_env = bob.make_env()
# init_pos = alice_env.initial_object_pos
# init_quat = alice_env.initial_object_quat

# print('goal pos: ', init_pos)
# print('goal quat: ', init_quat)
# bob_env.set_initial_state_and_goal_pose(init_pos, init_quat, init_pos, init_quat)
# obs = bob_env.reset()

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
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy, policy
from ray.rllib.utils.typing import AgentID, PolicyID
from env.multiagent_env import AsymMultiAgent
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.tune import register_env
from multi_trajectory_collector import MultiTrajectoryCollector
import ray.rllib.agents.ppo as ppo
from gym import spaces
import numpy as np
import time
from typing import Dict, Optional, TYPE_CHECKING


# class MyCallbacks(DefaultCallbacks):
#     def on_sample_end(self, *, worker: "RolloutWorker", samples: SampleBatch, **kwargs) -> None:
#         print('Type of sample batch: ', type(samples))
#         print('SAMPLES: ', samples)
#         print("returned sample batch of size {}".format(samples.count))

#     def on_episode_start(self, *, worker: "RolloutWorker", base_env: BaseEnv, policies: Dict[PolicyID, Policy], episode: Episode, **kwargs) -> None:
#         print('STARTING EPISODE')
#         print('POLICIES STARTIN EPISODE: ', policies)

#     def on_episode_end(self, *, worker: "RolloutWorker", base_env: BaseEnv, policies: Dict[PolicyID, Policy], episode: Episode, **kwargs) -> None:
#         print('EPISODE END')
#         print('POLICIES ON EPISODE END: ', policies)

#     def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Optional[Dict[PolicyID, Policy]] = None, episode: Episode, **kwargs) -> None:
        
#         print('STEP: ', base_env.get_sub_environments()[0].alice_step)
    
#     def on_postprocess_trajectory(self, *, worker: "RolloutWorker", episode: Episode, agent_id: AgentID, policy_id: PolicyID, policies: Dict[PolicyID, Policy], postprocessed_batch: SampleBatch, original_batches: Dict[AgentID, SampleBatch], **kwargs) -> None:
#         print('ON POSTPROCESS TRAJECTORY')
        # print(original_batches['alice'][1]['advantages'])
        # print(postprocessed_batch)
register_env("asym_self_play",
                 lambda _: AsymMultiAgent(
                     alice_steps=100, bob_steps=200, n_objects=1
                 ))

observation_space = spaces.Box(low=np.array([-6.5]*6), high=np.array([6.5]*6),dtype=np.float32)
action_space = spaces.MultiDiscrete(np.array([11]*6))

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    if agent_id == "alice":
        return "alice_policy"
    else:
        return "bob_policy"

config = {
    "num_workers": 1,
    "num_envs_per_worker": 1,
    "batch_mode": "complete_episodes",
    "framework": "torch",
    "multiagent": {
        "policies": {
            "alice_policy": (PPOTorchPolicy, observation_space, action_space, {}),
            "bob_policy": (PPOTorchPolicy, observation_space, action_space, {}),
        },
        "policy_mapping_fn": policy_mapping_fn,
        "policies_to_train": None,
    },
    # "callbacks": MyCallbacks,
    "sample_collector": MultiTrajectoryCollector,

}
# Create our RLlib Trainer.
trainer = ppo.PPOTrainer(env="asym_self_play", config=config)

t1 = time.time()
for _ in range(3):
    results = trainer.train()
    print('TRAIN FINISHED')

t2 = time.time()

