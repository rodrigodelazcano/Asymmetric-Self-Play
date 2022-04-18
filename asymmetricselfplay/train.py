from asymmetricselfplay.policies.bob_policy import BobTorchPolicy
from asymmetricselfplay.models.model import AsymModel
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
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray import tune
import ray
import argparse
import itertools
import os

prior_alice_policies_names = ["prior_alice_policy_" + str(i+1) for i in range(4)]
prior_bob_policies_names = ["prior_bob_policy_" + str(i+1) for i in range(4)]

def prior_alice_policy_mapping_fn(agent_id, episode, worker, **kwargs):
    if agent_id == "bob":
        return "bob_policy"
    else:
        return np.random.choice(["alice_policy", "prior_alice_policy_1", 
                                    "prior_alice_policy_2", "prior_alice_policy_3", "prior_alice_policy_4"],1,
                                    p=[.8, .1/2, .1/2, .1/2, .1/2])[0] 

def prior_bob_policy_mapping_fn(agent_id, episode, worker, **kwargs):
    if agent_id == "alice":
        return "alice_policy"
    else:
        return np.random.choice(["bob_policy", "prior_bob_policy_1", 
                                    "prior_bob_policy_2", "prior_bob_policy_3", "prior_bob_policy_4"],1,
                                    p=[.8, .1/2, .1/2, .1/2, .1/2])[0]

class AsymSelfPlayCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()

        self._alice_prior_pol_iter = itertools.cycle(prior_alice_policies_names)
        self._bob_prior_pol_iter = itertools.cycle(prior_bob_policies_names)
    
    def on_train_result(self, *, trainer, result, **kwargs):
        # Only one agent is allowed to compete against prior versions of its oponent per training iteration
        # Modify policy mapping function for each iteration
        if result["training_iteration"] % 2 == 0:
            policy_mapping_function = prior_bob_policy_mapping_fn
        else:
            policy_mapping_function = prior_alice_policy_mapping_fn
        def _set(worker):
            worker.set_policy_mapping_fn(policy_mapping_function)
        
        trainer.workers.foreach_worker(_set)

        # Update one prior alice + bob policy with current policy weights
        if result["training_iteration"] % 10 == 0:
            alice_prior_pol = self._alice_prior_pol_iter.__next__()
            bob_prior_pol = self._bob_prior_pol_iter.__next__()

            current_alice_state = trainer.get_policy("alice_policy").get_state()
            current_bob_state = trainer.get_policy("bob_policy").get_state()
            pol_map = trainer.workers.local_worker().policy_map
            pol_map[alice_prior_pol].set_state(current_alice_state)
            pol_map[bob_prior_pol].set_state(current_bob_state)

            trainer.workers.sync_weights(policies=[alice_prior_pol, bob_prior_pol])

parser = argparse.ArgumentParser()
parser.add_argument("--wandb-mode", type=str, default="disabled")
parser.add_argument("--alice-num-steps-per-goal", type=int, default=100)
parser.add_argument("--bob-num-steps-per-goal", type=int, default=200)
parser.add_argument("--num-objects", type=int, default=2)
parser.add_argument("--num-workers", type=int, default=1)
parser.add_argument("--num-envs-per-worker", type=int, default=1)
parser.add_argument("--num-gpus", type=int, default=0)
parser.add_argument("--rollout-fragment-lenght", type=int, default=5000)
parser.add_argument("--sgd-minibatch-size", type=int, default=4096)

# == Observation dict keys ==
# robot_state_keys = ["robot_joint_pos", "gripper_pos"]
# obj_state_keys = ["obj_pos", "obj_rot", "obj_vel_pos", "obj_vel_rot", "obj_rel_pos", "obj_gripper_contact"]
# goal_state_keys = ["goal_obj_pos", "goal_obj_rot", "rel_goal_obj_pos", "rel_goal_obj_rot"]


def get_rllib_configs():
    args = parser.parse_args()
    os.environ['WANDB_MODE'] = args.wandb_mode
    register_env("asym_self_play",
                 lambda _: AsymMultiAgent(
                     alice_steps=args.alice_num_steps_per_goal, 
                     bob_steps=args.bob_num_steps_per_goal,
                     n_objects=args.num_objects
                 ))

    ModelCatalog.register_custom_model("asym_torch_model", AsymModel)

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
        # Choose 
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
                    "dict_obs_space": observation_spaces[agent],
                    # == MLP ==
                    "mlp_hiddens": [512] * 5,
                    "mlp_activation": "relu",
                },
                # == LSTM ==
                # Max seq len for training the LSTM, defaults to 20.
                "max_seq_len": 20,
                # Size of the LSTM cell.
                "lstm_cell_size": 1024,
                # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
                "lstm_use_prev_action": True,
                # Whether to feed r_{t-1} to LSTM.
                "lstm_use_prev_reward": True if agent == "alice" else False,
                # Whether the LSTM is time-major (TxBx..) or batch-major (BxTx..).
                "_time_major": False,
            },
            "gamma": 0.998,
            "lr": 3e-4,
            "lambda": 0.95,
            "entropy_coeff": 0.01,
            "clip_param": 0.2,
            "ABC_loss_weight": 0.5 if agent=="bob" else None,
        } for agent in agents
    }

    policies_dict = {"alice_policy": policy.PolicySpec(policy_class=PPOTorchPolicy,
                                                    observation_space=observation_spaces["alice"],
                                                    action_space=action_space,
                                                    config=policy_configs["alice"]),
                     "bob_policy": policy.PolicySpec(policy_class=BobTorchPolicy, 
                                                observation_space=observation_spaces["bob"], 
                                                action_space=action_space, 
                                                config=policy_configs["bob"])}
    alice_prior_policies = {pol: policy.PolicySpec(policy_class=PPOTorchPolicy,
                                                    observation_space=observation_spaces["alice"],
                                                    action_space=action_space,
                                                    config=policy_configs["alice"]) for pol in prior_alice_policies_names}
    
    bob_prior_policies = {pol: policy.PolicySpec(policy_class=BobTorchPolicy, 
                                                    observation_space=observation_spaces["bob"], 
                                                    action_space=action_space, 
                                                    config=policy_configs["bob"]) for pol in prior_bob_policies_names}
   
    policies_dict.update(alice_prior_policies)
    policies_dict.update(bob_prior_policies)

    config = {
        "env": "asym_self_play",
        "callbacks": AsymSelfPlayCallback,
        "num_workers": args.num_workers,
        "num_envs_per_worker": args.num_envs_per_worker,
        "num_gpus": args.num_gpus,
        "rollout_fragment_length": args.rollout_fragment_lenght,
        "batch_mode": "complete_episodes",
        "framework": "torch",
        "train_batch_size": args.rollout_fragment_lenght*args.num_workers*args.num_envs_per_worker,
        "sgd_minibatch_size": args.sgd_minibatch_size,
        # sample reuse or epochs per training iterations
        "num_sgd_iter": 3,
        "multiagent": {
            "policies": policies_dict,
            "policy_mapping_fn": policy_mapping_fn,
            "policies_to_train": ["alice_policy", "bob_policy"],
        },
        "sample_collector": MultiEpisodeCollector,
    }   

    stop = {
            "training_iteration": 50000,
            "timesteps_total": 1000000,
            "episode_reward_mean": 7.0,
        }

    return config, stop
       
def run_tune():
    config, stop = get_rllib_configs()
    tune_analysis = tune.run("PPO", config=config, stop=stop, checkpoint_freq=10, checkpoint_at_end=True, callbacks=[WandbLoggerCallback(
            project="AsymmetricSelfPlay",
            api_key_file="wandb_api_key",
            log_config=True,
            )])

    return tune_analysis

if __name__ == "__main__":
    ray.init()
    results = run_tune() 
    ray.shutdown()
