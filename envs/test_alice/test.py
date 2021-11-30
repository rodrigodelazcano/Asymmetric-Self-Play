import alice
from robogym.envs.rearrange import blocks

# env2 = blocks.make_env()
# env2.reset()

env1 = alice.make_env()
env1.reset()

while True:

    action = env1.action_space.sample()
    env1.step(action)
    env1.render()


