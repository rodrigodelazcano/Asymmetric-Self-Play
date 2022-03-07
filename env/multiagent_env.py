from matplotlib.pyplot import axis
from . import bob
from .alice import alice
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np


class AsymMultiAgent(MultiAgentEnv):
    def __init__(self, alice_steps=100, bob_steps=100, n_objects=1) -> None:
        
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
                'max_goal_setting_steps_per_object': alice_steps
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
                        'object_out_placement_area': 3.0,
                        'objects_did_not_move': 0.0,
                    }
                }
            },
            constants={
                'max_timesteps_per_goal_per_obj': bob_steps
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
    
    def reset(self):
        self.bob_done = False
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
            info_d = {agent: {"build_next_batch": False}}
            rew_d[agent] = reward
            self.done_d[agent] = done
            obs_d[agent] = self._generate_observation_dictionary(obs, agent)

            if agent == "alice" and done:
                # if info["valid_goal"]:
                # If bob is not done 
                self.goal_setting += 1
                print('GOAL SETTING: ', self.goal_setting)
                if not self.bob_done:
                    init_pos = self.envs[agent].initial_object_pos
                    init_quat = self.envs[agent].initial_object_quat
                    goal_pos = info["last_object_pos"]
                    goal_quat = info["last_object_quat"]

                    # set alice's initial and goal object pose in bob's environment
                    self.envs["bob"].set_initial_state_and_goal_pose(init_pos, init_quat, goal_pos, goal_quat)
                    obs = self.envs["bob"].reset()
                    obs_d["bob"] = self._generate_observation_dictionary(obs, "bob")
                    
                    self.done_d['bob'] = False
                    # info_d = {'bob': {"new_traj": True}, 'alice': {"is_bob_done": self.bob_done}}
                # If bob is done for the rest of the episode because of incompleted goal
                else:
                    # Only iterate 5 times for goal setting
                    self.done_d["bob"] = True
                    self.done_d["alice"] = True
                    self.done_d["__all__"] = True
                    if self.goal_setting >= 5:
                        info_d = {agent: {"build_next_batch": True}}
                        self.goal_setting = 0
                        # info_d = [agent]["build_next_batch"] = True
                    # Bob is done, generate new trajectory for alice
                    # else:
                    #     obs_d[agent] = self.envs[agent].reset()
                    #     self.done_d["alice"] = False
                # If the final goal set by alice is not valid, end complete episode
                # else:
                #     print('not valid goal')
                #     self.done_d["bob"] = True
                #     self.done_d["alice"] = True
                #     self.done_d["__all__"] = True

            # When bob's trajectory is done
            elif agent == "bob" and done:
                # print('bob done')
                # if not obs['is_goal_achieved'][0]:
                #     rew_d["alice"] = 5
                #     self.bob_done = True
                #     # info_d["bob"] = info                    
                self.done_d["bob"] = True
                self.done_d["alice"] = True
                self.done_d["__all__"] = True
                if self.goal_setting >= 5:
                    self.goal_setting = 0
                    info_d = {agent: {"build_next_batch": True}}
                    
                # else:
                # self.done_d['alice'] = True
                # self.done_d["__all__"] = True
                # info_d = {'alice': {"new_traj": True, "is_bob_done": self.bob_done}}
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
                    
