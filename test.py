from env.alice import alice

alice_env = alice.make_env(
    
    parameters={
        'simulation_params': {
            'num_objects': 2,
            'max_num_objects': 8,
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
done = False
while not done:

    action = alice_env.action_space.sample()
    obs, reward, done, info = alice_env.step(action)
    alice_env.render()
    


