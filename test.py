from env import bob
from env.alice import alice

alice_env = alice.make_env(
    
    parameters={
        'simulation_params': {
            'num_objects': 1,
            'penalty': {
                'table_collision': 0.0, 
                'objects_off_table': 0.0,
                'wrist_collision': 0.0,
                'object_out_placement_area': 3.0,
                'objects_did_not_move': 0.0,
            }
        }
    }
)

bob_env = bob.make_env(
    
    parameters={
        'simulation_params': {
            'num_objects': 1,
            'penalty': {
                'table_collision': 0.0, 
                'objects_off_table': 0.0,
                'wrist_collision': 0.0,
                'object_out_placement_area': 3.0,
                'objects_did_not_move': 0.0,
            }
        }
    }
)
alice_env.reset()
bob_env.reset()
done = False


while not done:

    action = bob_env.action_space.sample()
    obs, reward, done, info = bob_env.step(action)
    bob_env.render()

    action = alice_env.action_space.sample()
    alice_env.step(action)
    alice_env.render()

    # print('obs: ', obs)
    # print('reward: ', reward)
    # print('done: ', done)
    # print('info: ', info)
    


