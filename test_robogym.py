from envs.alice import alice
from robogym.envs.rearrange import blocks
env1 = blocks.make_env()

obs = env1.reset()

print('testing alice now')
env2 = alice.make_env()

env2.reset()
