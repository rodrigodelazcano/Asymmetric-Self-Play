from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.evaluation.worker_set import WorkerSet
from env.multiagent_env import AsymMultiAgent
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.agents.pg.pg_tf_policy import PGTFPolicy
# from ray.rllib.examples. import M
# import random
import numpy as np
from gym import spaces
# from gym.spaces import Box, Discrete

observation_space = spaces.Box(low=np.array([-6.5]*6), high=np.array([6.5]*6),dtype=np.float32)
action_space = spaces.MultiDiscrete(np.array([11]*6))

# worker = RolloutWorker(env_creator=lambda _: AsymMultiAgent(
#                      alice_steps=20, bob_steps=200, n_objects=1)
#                  , policy_spec={
#                      "alice":
#                      (PPOTorchPolicy, observation_space, action_space),
#                      "bob":
#                      (PPOTorchPolicy, observation_space, action_space)
#                  },
#                  policy_mapping_fn=lambda agent_id:
#                 "alice"  # Traffic lights are always controlled by this policy
#                 if agent_id.startswith("alice")
#                 else "bob"  # Randomly c hoose from car policies
#                 , policy_config= {
#                     "framework": "torch"
#                 },
#                 batch_mode = "complete_episodes"
#                 )


# sample = worker.sample()


# workers = self._make_workers(
#                 env_creator=env_creator,
#                 validate_env=validate_env,
#                 policy_class=self._policy_class,
#                 config=config,
#                 num_workers=self.config["num_workers"])

# WorkerSet(
#             env_creator=env_creator,
#             validate_env=validate_env,
#             policy_class=policy_class,
#             trainer_config=config,
#             num_workers=num_workers,
#             logdir=self.logdir)