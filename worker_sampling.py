from ray.rllib.evaluation.rollout_worker import RolloutWorker
from env.multiagent_env import AsymMultiAgent
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
import numpy as np
from gym import spaces

observation_space = spaces.Box(low=np.array([-6.5]*6), high=np.array([6.5]*6),dtype=np.float32)
action_space = spaces.MultiDiscrete(np.array([11]*6))

worker = RolloutWorker(env_creator=lambda _: AsymMultiAgent(
                     alice_steps=100, bob_steps=200, n_objects=1)
                 , policy_spec={
                     "alice":
                     (PPOTorchPolicy, observation_space, action_space),
                     "bob":
                     (PPOTorchPolicy, observation_space, action_space)
                 },
                 policy_mapping_fn=lambda agent_id:
                "alice"  # Traffic lights are always controlled by this policy
                if agent_id.startswith("alice")
                else "bob"  # Randomly c hoose from car policies
                , policy_config= {
                    "framework": "torch"
                }
                )


sample = worker.sample()
print(sample)