from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch


batch = SampleBatch({'actions':[1,2,3],'rewards':[1,2,3]})

print(batch['rewards'])

multiagent_batch = MultiAgentBatch({'alice':batch,'bob':batch},env_steps=100)