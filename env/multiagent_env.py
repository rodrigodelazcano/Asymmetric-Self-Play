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
        self.dones = dict.fromkeys(self.agents, False)

        self.episode = 0
    
    def reset(self):
        self.episode += 1
        self.goal_setting = 0
        self.dones = dict.fromkeys(self.agents, False)
        alice_init_obs = self.envs["alice"].reset()['robot_joint_pos']
        # Only reset alice env. Trainer computes actions only for agents in the observation dict.
        return {"alice": alice_init_obs}
    
    def step(self, action_dict):
        obs_d = {}
        rew_d = {}
        done_d = {}
        info_d = {}

        done_d["__all__"] = False
        if not action_dict:
            obs = self.envs["alice"].reset()['robot_joint_pos']
            return obs_d, rew_d, done_d, info_d       
        for agent, action in action_dict.items():
            obs, reward, done, info = self.envs[agent].step(action)
            rew_d[agent] = reward

            if agent == "alice" and done:              
                if info["valid_goal"]:
                    self.goal_setting += 1
                    init_pos = self.envs[agent].initial_object_pos
                    init_quat = self.envs[agent].initial_object_quat
                    goal_pos = info["last_object_pos"]
                    goal_quat = info["last_object_quat"]
                    
                    # do only if bob is not done
                    if not self.dones["bob"]:
                        # set alice's initial and goal object pose in bob's environment
                        self.envs["bob"].set_initial_state_and_goal_pose(init_pos, init_quat, goal_pos, goal_quat)
                        obs = self.envs["bob"].reset()
                        obs_d["bob"] = obs['robot_joint_pos']
                    else:
                        if self.goal_setting >= 5:
                            done_d["bob"] = True
                            done_d["alice"] = True
                            done_d["__all__"] = True
                        else:
                            obs_d[agent] = self.envs[agent].reset()['robot_joint_pos']
                            # info_d[agent] = info
            
                    return obs_d, rew_d, done_d, info_d

                else:
                    done_d["bob"] = True
                    done_d["alice"] = True
                    done_d["__all__"] = True        
            elif agent == "bob" and done:
                if not obs['is_goal_achieved'][0]:
                    rew_d["alice"] = 5
                    self.dones["bob"] = True
                    done_d["bob"] = True
                    # info_d["bob"] = info
                    return obs_d, rew_d, done_d, info_d

                if self.goal_setting >= 5:
                            done_d["bob"] = True
                            done_d["alice"] = True
                            done_d["__all__"] = True
                    # What happens if I return an empty obs ? 
            
            obs_d[agent] = obs['robot_joint_pos']
            # info_d[agent] = info

        return obs_d, rew_d, done_d, info_d
                    
   
    