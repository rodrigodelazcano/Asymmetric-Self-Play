from policies.bob_policy import BobTorchPolicy
from policies.model import AsymModel
from ray.rllib.policy import policy
from ray.rllib.models import ModelCatalog
from env.multiagent_env import AsymMultiAgent
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.tune import register_env
from multi_episode_collector import MultiEpisodeCollector
from gym import spaces
import numpy as np
import os
from collections import OrderedDict
from ray.tune.integration.wandb import WandbLoggerCallback, WandbLogger
from ray import tune
import ray

number_of_objects = 2

register_env("asym_self_play",
                 lambda _: AsymMultiAgent(
                     alice_steps=103, bob_steps=200, n_objects=number_of_objects
                 ))

ModelCatalog.register_custom_model("asym_torch_model", AsymModel)

robot_state_keys = ["robot_joint_pos", "gripper_pos"]
obj_state_keys = ["obj_pos", "obj_rot", "obj_vel_pos", "obj_vel_rot", "obj_rel_pos", "obj_gripper_contact"]
goal_state_keys = ["goal_obj_pos", "goal_obj_rot", "rel_goal_obj_pos", "rel_goal_obj_rot"]

observation_spaces = {}
observation_space_bob = OrderedDict({
    "robot_joint_pos": spaces.Box(low=np.array([-6.5]*6), high=np.array([6.5]*6),dtype=np.float32),
    "gripper_pos": spaces.Box(low=np.array([-np.inf]*3), high=np.array([np.inf]*3),dtype=np.float32),
    })
observation_space_alice = OrderedDict({
    "robot_joint_pos": spaces.Box(low=np.array([-6.5]*6), high=np.array([6.5]*6),dtype=np.float32),
    "gripper_pos": spaces.Box(low=np.array([-np.inf]*3), high=np.array([np.inf]*3),dtype=np.float32),
    })

for i in range(number_of_objects):
    observation_space_bob["obj_"+str(i)+"_state"] = \
        spaces.Box(low=np.array([-np.inf]*29), high=np.array([np.inf]*29), dtype=np.float32)

for i in range(number_of_objects):
    observation_space_alice["obj_"+str(i)+"_state"] = \
        spaces.Box(low=np.array([-np.inf]*17), high=np.array([np.inf]*17), dtype=np.float32)

observation_spaces["alice"] = spaces.Dict(sorted(observation_space_alice.items()))
observation_spaces["bob"] = spaces.Dict(sorted(observation_space_bob.items()))


# observation_space = spaces.Box(low=np.array([-6.5]*6), high=np.array([6.5]*6),dtype=np.float32)
action_space = spaces.MultiDiscrete(np.array([11]*6))

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    if agent_id == "alice":
        return "alice_policy"
    else:
        return "bob_policy"

# def observation_fn(agent_obs, worker, base_env, policies, episode):
#     return agent_obs

agents = ["alice", "bob"]

ray.init()

policy_configs = {
    agent: {
        "model": {
            "custom_model": "asym_torch_model",
            "custom_model_config": {
                "number_of_objects": number_of_objects,
                "num_model_outputs": 256,
                "dict_obs_space": observation_spaces[agent],
            },
            # # == LSTM ==
            # # Whether to wrap the model with an LSTM.
            # "use_lstm": False,
            # # Max seq len for training the LSTM, defaults to 20.
            "max_seq_len": 20,
            # # Size of the LSTM cell.
            "lstm_cell_size": 256,
            # # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
            "lstm_use_prev_action": True,
            # # Whether to feed r_{t-1} to LSTM.
            "lstm_use_prev_reward": True if agent == "alice" else False,
            # # Whether the LSTM is time-major (TxBx..) or batch-major (BxTx..).
            "_time_major": False,
            # "_disable_preprocessor_api": True,
        },
        "beta": 0.5 if agent=="bob" else None
    } for agent in agents
}

config = {
    "env": "asym_self_play",
    "num_workers": 1,
    "num_envs_per_worker": 1,
    "rollout_fragment_length": 1000,
    "batch_mode": "complete_episodes",
    "framework": "torch",
    "train_batch_size": 2000,
    "sgd_minibatch_size": 60,
    "logger_config": {
            "wandb": {
                "project": "Asymmetric_Self_Play",
                "api_key": "1f77142634341e49c67a4f09fffb3bd79abc4f71",
            }
    },
    "multiagent": {
        "policies": {
            "alice_policy": policy.PolicySpec(policy_class=PPOTorchPolicy,
                                                observation_space=observation_spaces["alice"],
                                                action_space=action_space,
                                                config=policy_configs["alice"]),
            "bob_policy": policy.PolicySpec(policy_class=BobTorchPolicy, 
                                                observation_space=observation_spaces["bob"], 
                                                action_space=action_space, 
                                                config=policy_configs["bob"]
                                            ),
        },
        "policy_mapping_fn": policy_mapping_fn,
    },
    "sample_collector": MultiEpisodeCollector,
}   

stop = {
        "training_iteration": 5,
        "timesteps_total": 100000,
        "episode_reward_mean": 200.0,
    }

results = tune.run("PPO", config=config, stop=stop, checkpoint_freq=1, checkpoint_at_end=True, loggers=[WandbLogger])

ray.shutdown()
