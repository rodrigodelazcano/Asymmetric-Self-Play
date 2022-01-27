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
from policies.bob_policy import BobTorchPolicy
from policies.bob_model import BobModel
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy, policy
from ray.rllib.utils.typing import AgentID, PolicyID
from ray.rllib.models import ModelCatalog
from env.multiagent_env import AsymMultiAgent
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.tune import register_env
from multi_trajectory_collector import MultiTrajectoryCollector
import ray.rllib.agents.ppo as ppo
from gym import spaces
import numpy as np
import time
from typing import Dict, Optional, TYPE_CHECKING


class MyCallbacks(DefaultCallbacks):

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Optional[Dict[PolicyID, Policy]] = None, episode: Episode, **kwargs) -> None:
        # when new trajectory in agent set last_done to False (because it revives)
        for env_state in base_env.env_states:
            all_agents_done = env_state.last_dones['__all__']
            last_done_keys = env_state.last_dones.keys()
            if not all_agents_done:
                for ag in last_done_keys:
                    if ag != '__all__' and env_state.last_dones[ag]:
                        env_state.last_dones[ag] = False

number_off_objects = 1

register_env("asym_self_play",
                 lambda _: AsymMultiAgent(
                     alice_steps=100, bob_steps=200, n_objects=number_off_objects
                 ))

ModelCatalog.register_custom_model("bob_torch_model", BobModel)

# observation_space = {}
# observation_space_bob = spaces.Dict({
#     "robot_joint_pos": spaces.Box(low=np.array([-6.5]*6), high=np.array([6.5]*6),dtype=np.float32),
#     "gripper_pos": spaces.Box(low=np.array([-np.inf]*3), high=np.array([np.inf]*3)),
    # "obj_state": spaces.Box(low=np.array([-np.inf]*17), high=np.array([np.inf]*17)),
    # "goal_state": spaces.Box(low=np.array([-np.inf]*12), high=np.array([np.inf]*12))
# })
observation_space = spaces.Box(low=np.array([-6.5]*6), high=np.array([6.5]*6),dtype=np.float32)
action_space = spaces.MultiDiscrete(np.array([11]*6))

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    if agent_id == "alice":
        return "alice_policy"
    else:
        return "bob_policy"

bob_config = {
    "model": {
        "custom_model": "bob_torch_model"
    }
}

config = {
    "num_workers": 1,
    "num_envs_per_worker": 1,
    "batch_mode": "complete_episodes",
    "framework": "torch",
    "train_batch_size": 400,
    "sgd_minibatch_size": 50,
    "multiagent": {
        "policies": {
            "alice_policy": (PPOTorchPolicy, observation_space, action_space, {}),
            # "bob_policy": policy.PolicySpec(policy_class=BobTorchPolicy, 
            #                                     observation_space = observation_space_bob, 
            #                                     action_space = action_space, 
            #                                     config=bob_config
            #                                 ),
            "bob_policy": (PPOTorchPolicy, observation_space, action_space, {}),
        },
        "policy_mapping_fn": policy_mapping_fn,
        "policies_to_train": None,
    },
    "callbacks": MyCallbacks,
    "sample_collector": MultiTrajectoryCollector,
}   

# Create our RLlib Trainer.
trainer = ppo.PPOTrainer(env="asym_self_play", config=config)
print('HELLOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO')
print('CONFIG: ',trainer.config["keep_per_episode_custom_metrics"])
t1 = time.time()
for _ in range(3):
    results = trainer.train()
    print(results)

t2 = time.time()

