from robogym.envs.rearrange import blocks_attached

env = blocks_attached.make_env(
    constants={
        'randomize': True,
        'mujoco_substeps': 10,
        'max_timesteps_per_goal': 400
    },
    parameters={
        'robot_control_params': {
            'control_mode': "tcp+roll+yaw"      # set the control type tcp + yaw + pitch
        },
        'simulation_params': {

        }
    })
# action space of the robot 6 dimensions with 11 bins
print('ACTION SPACE: ', env.action_space)

# observation space
print('OBSERVATION SPACE: ', env.observation_space)



# while(True):

#     # render the simulation
#     env.render()
