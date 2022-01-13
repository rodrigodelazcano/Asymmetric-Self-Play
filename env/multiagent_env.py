from . import bob
from .alice import alice
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class AsymMultiAgent(MultiAgentEnv):
    def __init__(self, alice_steps, bob_steps, n_objects) -> None:
        
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

        self.envs = dict(zip(self.agents, envs))

        self.goal_setting = 0
        self.done_d = dict.fromkeys(self.agents, False)
        self.done_d["__all__"] = False

        self.episode = 0

        self.alice_step = 0
    
    def reset(self):
        self.episode += 1
        self.goal_setting = 0
        alice_init_obs = self.envs["alice"].reset()['robot_joint_pos']
        self.done_d = dict.fromkeys(self.done_d.keys(), False)
        print('SELF DONES D: ', self.done_d)
        # Only reset alice env. Trainer computes actions only for agents in the observation dict.
        return {"alice": alice_init_obs}
    
    def step(self, action_dict):
        obs_d = {}
        rew_d = {}
        info_d = {}
        
        # Start new multi-goal cycle
        if not action_dict:
            self.done_d["alice"] = False
            obs = self.envs["alice"].reset()['robot_joint_pos']
            obs_d['alice'] = obs
            self.alice_step += 1
            return obs_d, rew_d, self.done_d, info_d   

        for agent, action in action_dict.items():
            obs, reward, done, info = self.envs[agent].step(action)
            rew_d[agent] = reward
            self.done_d[agent] = done
            info_d[agent] = {"new_traj": None}

            if agent == "alice" and done: 
                # if info["valid_goal"]:
                self.goal_setting += 1
                init_pos = self.envs[agent].initial_object_pos
                init_quat = self.envs[agent].initial_object_quat
                goal_pos = info["last_object_pos"]
                goal_quat = info["last_object_quat"]
            
                # do only if bob is not done
                print('IS BOB DONE: ', self.done_d['bob'])
                if not self.done_d["bob"]:
                    # set alice's initial and goal object pose in bob's environment
                    self.envs["bob"].set_initial_state_and_goal_pose(init_pos, init_quat, goal_pos, goal_quat)
                    obs = self.envs["bob"].reset()
                    obs_d["bob"] = obs['robot_joint_pos']
                    info_d = {'bob': {"new_traj": True}}
                else:
                    if self.goal_setting >= 5:
                        self.done_d["bob"] = True
                        self.done_d["alice"] = True
                        self.done_d["__all__"] = True
                    else:
                        obs_d[agent] = self.envs[agent].reset()['robot_joint_pos']
                        info_d[agent] = {}
                info_d = {}
                return obs_d, rew_d, self.done_d, info_d

                # else:
                #     self.done_d["bob"] = True
                #     self.done_d["alice"] = True
                #     self.done_d["__all__"] = True        
            elif agent == "bob" and done:
                print('BOB GOAL ACHIEVED: ', obs['is_goal_achieved'])
                if not obs['is_goal_achieved'][0]:
                    rew_d["alice"] = 5
                    self.done_d["bob"] = True
                    # info_d["bob"] = info                    
                    

            if self.goal_setting >= 5:
                self.done_d["bob"] = True
                self.done_d["alice"] = True
                self.done_d["__all__"] = True
                    # What happens if I return an empty obs ?
                return obs_d, rew_d, self.done_d, info_d

            obs_d[agent] = obs['robot_joint_pos']
            # info_d[agent] = info
        return obs_d, rew_d, self.done_d, info_d
                    
   
    