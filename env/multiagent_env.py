from matplotlib.pyplot import axis
from . import bob
from .alice import alice
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np


class AsymMultiAgent(MultiAgentEnv):
    def __init__(self, alice_steps=100, bob_steps=100, n_objects=1) -> None:
        
        succes_threshold = {"obj_pos": 0.04, "obj_rot": 0.2}
        move_threshold = {"obj_pos": succes_threshold["obj_pos"] + 0.1, "obj_rot": succes_threshold["obj_pos"] + 0.05}
        alice_env = alice.make_env( 
            parameters={
                'simulation_params': {
                    'num_objects': n_objects,
                    'penalty': {
                        'table_collision': 0.0, 
                        'objects_off_table': 0.0,
                        'wrist_collision': 0.0,
                        'object_out_placement_area': 3.0,
                        'objects_did_not_move': 0.0,
                    }
                }
            },
            constants={
                'max_goal_setting_steps_per_object': alice_steps,
                'move_threshold': move_threshold,
            }
        )

        bob_env = bob.make_env(
            
            parameters={
                'simulation_params': {
                    'num_objects': n_objects,
                    'penalty': {
                        'table_collision': 0.0, 
                        'objects_off_table': 0.0,
                        'wrist_collision': 0.0,
                        'object_out_placement_area': 0.0,
                        'objects_did_not_move': 0.0,
                    }
                }
            },
            constants={
                'max_timesteps_per_goal_per_obj': bob_steps,
                'success_threshold': succes_threshold,
                'successes_needed': 1,
            }
        )

        self.agents = ['alice', 'bob']
        envs = [alice_env, bob_env]

        self.n_obj = n_objects

        self.envs = dict(zip(self.agents, envs))

        self.goal_setting = 0
        self.done_d = dict.fromkeys(self.agents, False)
        self.done_d["__all__"] = False

        self.robot_state_keys = ["robot_joint_pos", "gripper_pos"]
        self.obj_state_keys = ["obj_pos", "obj_rot", "obj_vel_pos",
                                "obj_vel_rot", "obj_rel_pos", "obj_gripper_contact"]
        self.goal_state_keys = ["goal_obj_pos", "goal_obj_rot", "rel_goal_obj_pos", "rel_goal_obj_rot"]

        self.episode = 0

        self.alice_step = 0

        self.bob_done = False
    
    def reset(self):
        obs = self.envs["alice"].reset()
        alice_init_obs = self._generate_observation_dictionary(obs, "alice")
        self.done_d = dict.fromkeys(self.done_d.keys(), False)
        # Only reset alice env. Trainer computes actions only for agents in the observation dict.
        return {"alice": alice_init_obs}
    
    def step(self, action_dict):
        obs_d = {}
        rew_d = {}
        info_d = {}

        for agent, action in action_dict.items():
            obs, reward, done, info = self.envs[agent].step(action)

            if agent == 'bob':
                print('REWARD FOR BOB: ', reward)
            if not done:
                info_d = {agent: {"build_next_batch": False}}
                rew_d[agent] = reward
                obs_d[agent] = self._generate_observation_dictionary(obs, agent)

            elif agent == "alice" and done:
                print('ALICE DONE')
                # If the final goal set by alice is invalid (any object off the table, 
                # or no object has been moved), end complete episode
                # Do not perform Behavioral Cloning
                print('GOAL SETTING: ', self.goal_setting)
                if info["valid_goal"]:
                    print('ALICE SET VALID GOAL')
                    self.goal_setting += 1
                    reward += 1.0
                    # If Bob has solved previous goal it is not done.
                    # Keep solving goals if not done, stop solving goals and
                    # implement Behavioral Cloning if done.
                    if not self.bob_done:
                        # Store last Alice obs, rew, and info for when Bob's episode is done
                        # and update Alice's reward.
                        self.alice_last_obs = self._generate_observation_dictionary(obs, agent)
                        self.alice_last_rew = reward
                        self.alice_last_info = {"build_next_batch": False, "bob_is_done": self.bob_done}
                        self.alice_last_info.update(info)

                        init_pos = self.envs[agent].initial_object_pos
                        init_quat = self.envs[agent].initial_object_quat
                        goal_pos = info["last_object_pos"]
                        goal_quat = info["last_object_quat"]

                        # set alice's initial and goal object pose in bob's environment
                        self.envs["bob"].set_initial_state_and_goal_pose(init_pos, init_quat, goal_pos, goal_quat)
                        obs = self.envs["bob"].reset()
                        obs_d["bob"] = self._generate_observation_dictionary(obs, "bob")                        
                    else:
                        info_d = {agent: {"build_next_batch": False, "bob_is_done": self.bob_done}}
                        info_d[agent].update(info)
                        rew_d[agent] = reward
                        obs_d[agent] = self._generate_observation_dictionary(obs, agent) 
                        
                        self.done_d["bob"] = True
                        self.done_d["alice"] = True
                        self.done_d["__all__"] = True

                        # 5 goal setting episodes per multiepisode cycle
                        if self.goal_setting >= 5:
                            info_d[agent]["build_next_batch"] = True
                            self.goal_setting = 0
                            self.bob_done = False
                else:
                    print('ALICE DID NOT SET A VALID GOAL')
                    info_d = {agent: {"build_next_batch": True, "bob_is_done": False}}
                    info_d[agent].update(info)
                    rew_d[agent] = reward
                    obs_d[agent] = self._generate_observation_dictionary(obs, agent)
                    self.done_d["bob"] = True
                    self.done_d["alice"] = True
                    self.done_d["__all__"] = True
                    self.goal_setting = 0
                    self.bob_done = False

            # Bob's trajectory is done
            elif agent == "bob" and done:
                print('BOB DONE')
                info_d = {agent: {"build_next_batch": False}}
                info_d[agent].update(info)
                obs_d[agent] = self._generate_observation_dictionary(obs, agent)
                rew_d[agent] = reward
                rew_d["alice"] = self.alice_last_rew
                obs_d["alice"] = self.alice_last_obs
                info_d["alice"] = self.alice_last_info

                self.done_d["bob"] = True
                self.done_d["alice"] = True
                self.done_d["__all__"] = True
                
                # Bob was not able to solve the goal
                if not obs['is_goal_achieved'][0]:
                    rew_d["alice"] += 5
                    print('BOB DID NOT ACHIEVE GOAL')
                    self.bob_done = True
                else:
                    rew_d[agent] += 5
                
                if self.goal_setting >= 5:
                    self.goal_setting = 0
                    info_d[agent]["build_next_batch"] = True
                info_d[agent].update({"bob_is_done": self.bob_done})

                if info_d[agent]["build_next_batch"]:
                    self.goal_setting = 0
                    print('BUILDING NEXT BATCH!!!!!')
                    self.bob_done = False

        return obs_d, rew_d, self.done_d, info_d
    
    def _generate_observation_dictionary(self, obs, agent):
        obs_dict = {key : obs[key] for key in ["robot_joint_pos", "gripper_pos"]}
        obj_state = np.concatenate(tuple([obs[key] for key in self.obj_state_keys]), axis=1)
        if agent == "bob":
            goal_state = np.concatenate(tuple([obs[key] for key in self.goal_state_keys]), axis=1)
            obj_state = np.concatenate((obj_state, goal_state), axis=1)
        
        for i in range(obj_state.shape[0]):
            obs_dict["obj_"+ str(i)+"_state"] = obj_state[i,:]

        return obs_dict
                    
