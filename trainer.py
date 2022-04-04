from matplotlib.cbook import contiguous_regions
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
from collections import OrderedDict
from ray.tune.integration.wandb import WandbLoggerCallback
from ray import tune
import ray
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num-objects", type=int, default=2)
parser.add_argument("--num-workers", type=int, default=4)



def get_rllib_configs():
    args = parser.parse_args()
    register_env("asym_self_play",
                 lambda _: AsymMultiAgent(
                     alice_steps=100, bob_steps=200, n_objects=args.num_objects
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

    for i in range(args.num_objects):
        observation_space_bob["obj_"+str(i)+"_state"] = \
            spaces.Box(low=np.array([-np.inf]*29), high=np.array([np.inf]*29), dtype=np.float32)

    for i in range(args.num_objects):
        observation_space_alice["obj_"+str(i)+"_state"] = \
            spaces.Box(low=np.array([-np.inf]*17), high=np.array([np.inf]*17), dtype=np.float32)

    observation_spaces["alice"] = spaces.Dict(sorted(observation_space_alice.items()))
    observation_spaces["bob"] = spaces.Dict(sorted(observation_space_bob.items()))

    action_space = spaces.MultiDiscrete(np.array([11]*6))

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id == "alice":
            return "alice_policy"
        else:
            return "bob_policy"

    agents = ["alice", "bob"]
    policy_configs = {
        agent: {
            "model": {
                "custom_model": "asym_torch_model",
                "custom_model_config": {
                    "number_of_objects": args.num_objects,
                    "num_model_outputs": 256,
                    "dict_obs_space": observation_spaces[agent],
                },
                # # == LSTM ==
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

            "gamma": 0.998,
            "lr": 3e-4,
            "lambda": 0.95,
            "entropy_coeff": 0.01,
            "clip_param": 0.2,
            "ABC_loss_weight": 0.5 if agent=="bob" else None,
        } for agent in agents
    }


    config = {
        "env": "asym_self_play",
        "num_workers": args.num_workers,
        "num_envs_per_worker": 1,
        "num_gpus": 0,
        "rollout_fragment_length": 4096,
        "batch_mode": "complete_episodes",
        "framework": "torch",
        "train_batch_size": 40960,
        "sgd_minibatch_size": 128,
        # sample reuse or epochs per training iterations
        "num_sgd_iter": 3,
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
            "training_iteration": 50000,
            "timesteps_total": 1000000,
            "episode_reward_mean": 200.0,
        }
    
    return config, stop
def run_tune():
    config, stop = get_rllib_configs()
    tune_analysis = tune.run("PPO", config=config, stop=stop, checkpoint_freq=500, checkpoint_at_end=True, callbacks=[WandbLoggerCallback(
            project="AsymmetricSelfPlay",
            api_key_file="wandb_api_key",
            log_config=False)])
    
    return tune_analysis


if __name__ == "__main__":
    ray.init()
    results = run_tune() 
    ray.shutdown()