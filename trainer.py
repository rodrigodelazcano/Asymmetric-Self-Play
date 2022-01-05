# from env import bob
# from env.alice import alice


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
from env.multiagent_env import AsymMultiAgent
from ray.tune import register_env
import ray
import ray.rllib.agents.ppo as ppo
from gym import spaces
import numpy as np

register_env("asym_self_play",
                 lambda _: AsymMultiAgent(
                     alice_steps=100, bob_steps=200, n_objects=1
                 ))

observation_space = spaces.Box(low=np.array([-6.5]*6), high=np.array([6.5]*6),dtype=np.float32)
action_space = spaces.MultiDiscrete(np.array([11]*6))

config = {
    "num_workers": 1,
    "batch_mode": "truncate_episodes",
    "framework": "torch",
    "multiagent": {
        "policies": {
            # the first tuple value is None -> uses default policy
            "alice": (None, observation_space, action_space, {}),
            "bob": (None, observation_space, action_space, {}),
        },
        "policy_mapping_fn":
            lambda agent_id:
                "alice"  # Traffic lights are always controlled by this policy
                if agent_id.startswith("alice")
                else "bob"  # Randomly c hoose from car policies
    },

}
# Create our RLlib Trainer.
trainer = ppo.PPOTrainer(env="asym_self_play", config=config)

for _ in range(3):
    print(trainer.train())
