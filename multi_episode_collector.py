from gym.spaces import Space
import logging
import numpy as np
from typing import TYPE_CHECKING, Union

from ray.rllib.evaluation.episode import Episode
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.debug import summarize
from ray.rllib.utils.typing import AgentID, EnvID, PolicyID, \
    TensorType
from ray.util.debug import log_once

if TYPE_CHECKING:
    from ray.rllib.agents.callbacks import DefaultCallbacks

logger = logging.getLogger(__name__)


from ray.rllib.evaluation.collectors.simple_list_collector import SimpleListCollector, _AgentCollector, _PolicyCollector, _PolicyCollectorGroup

class MultiEpisodeCollector(SimpleListCollector):
    def __init__(self, policy_map: PolicyMap, clip_rewards: Union[bool, float], callbacks: "DefaultCallbacks", multiple_episodes_in_batch: bool = True, rollout_fragment_length: int = 200, count_steps_by: str = "env_steps"):
        super().__init__(policy_map, clip_rewards, callbacks, multiple_episodes_in_batch, rollout_fragment_length, count_steps_by)

        self.policy_collector_groups = {}
    
    @override(SimpleListCollector)
    def add_init_obs(self, episode: Episode, agent_id: AgentID, env_id: EnvID,
                     policy_id: PolicyID, t: int,
                     init_obs: TensorType) -> None:
        # Make sure our mappings are up to date.
        agent_key = (episode.episode_id, agent_id)
        self.agent_key_to_policy_id[agent_key] = policy_id
        policy = self.policy_map[policy_id]
        view_reqs = policy.model.view_requirements if \
            getattr(policy, "model", None) else policy.view_requirements

        # Add initial obs to Trajectory.
        assert agent_key not in self.agent_collectors
        # TODO: determine exact shift-before based on the view-req shifts.
        self.agent_collectors[agent_key] = _AgentCollector(view_reqs, policy)
        self.agent_collectors[agent_key].add_init_obs(
            episode_id=episode.episode_id,
            agent_index=episode._agent_index(agent_id),
            env_id=env_id,
            t=t,
            init_obs=init_obs)

        self.episodes[episode.episode_id] = episode
        if episode.batch_builder is None:
            if self.policy_collector_groups.get(env_id):
                episode.batch_builder = self.policy_collector_groups.get(env_id).pop()
            else:
                episode.batch_builder = _PolicyCollectorGroup(self.policy_map)
                self.policy_collector_groups[env_id] = []

        self._add_to_next_inference_call(agent_key)

    @override(SimpleListCollector)
    def postprocess_episode(
            self,
            episode: Episode,
            is_done: bool = False,
            check_dones: bool = False,
            build: bool = False) -> Union[None, SampleBatch, MultiAgentBatch]:
        episode_id = episode.episode_id
        env_id = episode.env_id
        policy_collector_group = episode.batch_builder

        # TODO: (sven) Once we implement multi-agent communication channels,
        #  we have to resolve the restriction of only sending other agent
        #  batches from the same policy to the postprocess methods.
        # Build SampleBatches for the given episode.
        # for env_id in self.build_next_batch:
        pre_batches = {}

        for (eps_id, agent_id), collector in self.agent_collectors.items():
            if collector.agent_steps == 0 or eps_id != episode_id:
                continue
            pid = self.agent_key_to_policy_id[(eps_id, agent_id)]
            policy = self.policy_map[pid]
            pre_batch = collector.build(policy.view_requirements)
            pre_batches[agent_id] = (policy, pre_batch)
        # Apply reward clipping before calling postprocessing functions.
        if self.clip_rewards is True:
            for _, (_, pre_batch) in pre_batches.items():
                pre_batch["rewards"] = np.sign(pre_batch["rewards"])
        elif self.clip_rewards:
            for _, (_, pre_batch) in pre_batches.items():

                pre_batch["rewards"] = np.clip(
                    pre_batch["rewards"],
                    a_min=-self.clip_rewards,
                    a_max=self.clip_rewards)

        post_batches = {}
        relable_demonstration = False
        for agent_id, (_, pre_batch) in pre_batches.items():
            build_next_batch = pre_batch[SampleBatch.INFOS][-1]["build_next_batch"]       
            relable_demonstration = pre_batch[SampleBatch.INFOS][-1]["bob_is_done"]

            if is_done and check_dones and \
                    not pre_batch[SampleBatch.DONES][-1]:
                raise ValueError(
                    "Episode {} terminated for all agents, but we still "
                    "don't have a last observation for agent {} (policy "
                    "{}). ".format(
                        episode_id, agent_id, self.agent_key_to_policy_id[(
                            episode_id, agent_id)]) +
                    "Please ensure that you include the last observations "
                    "of all live agents when setting done[__all__] to "
                    "True. Alternatively, set no_done_at_end=True to "
                    "allow this.")

            if len(pre_batches) > 1:
                other_batches = pre_batches.copy()
                del other_batches[agent_id]
            else:
                other_batches = {}
            pid = self.agent_key_to_policy_id[(episode_id, agent_id)]
            policy = self.policy_map[pid]
            if any(pre_batch[SampleBatch.DONES][:-1]) or len(
                    set(pre_batch[SampleBatch.EPS_ID])) > 1:
                raise ValueError(
                    "Batches sent to postprocessing must only contain steps "
                    "from a single trajectory.", pre_batch)
            # Call the Policy's Exploration's postprocess method.
            post_batches[agent_id] = pre_batch
            if getattr(policy, "exploration", None) is not None:
                policy.exploration.postprocess_trajectory(
                    policy, post_batches[agent_id], policy.get_session())
            post_batches[agent_id].set_get_interceptor(None)
            post_batches[agent_id] = policy.postprocess_trajectory(
                post_batches[agent_id], other_batches, episode)
            # print('POST BATCH: ', post_batches[agent_id])

        if log_once("after_post"):
            logger.info(
                "Trajectory fragment after postprocess_trajectory():\n\n{}\n".
                format(summarize(post_batches)))

        # Append into policy batches and reset.
        from ray.rllib.evaluation.rollout_worker import get_global_worker
        for agent_id, post_batch in sorted(post_batches.items()):
            # if agent_id == "alice":
            #     print('ALICE POST BATCH: ', post_batch)
            #     print('ALICE POSTBATCH OBS TYPE: ', post_batch['rewards'])

            agent_key = (episode_id, agent_id)
            pid = self.agent_key_to_policy_id[agent_key]
            policy = self.policy_map[pid]
            self.callbacks.on_postprocess_trajectory(
                worker=get_global_worker(),
                episode=episode,
                agent_id=agent_id,
                policy_id=pid,
                policies=self.policy_map,
                postprocessed_batch=post_batch,
                original_batches=pre_batches)

            # Add the postprocessed SampleBatch to the policy collectors for
            # training.
            # Policy id (PID) may be a newly added policy. Just confirm we have it in our
            # policy map before proceeding with adding a new _PolicyCollector()
            # to the group.
            if pid not in policy_collector_group.policy_collectors:
                assert pid in self.policy_map
                policy_collector_group[
                    pid] = _PolicyCollector(policy)

            policy_collector_group.policy_collectors[
                pid].add_postprocessed_batch_for_training(
                    post_batch, policy.view_requirements)
            if is_done:
                del self.agent_key_to_policy_id[agent_key]
                del self.agent_collectors[agent_key]

        if relable_demonstration:
            alice_post_batch = policy_collector_group.policy_collectors["alice_policy"].batches[-1].copy()
            alice_observation_space = self.policy_map["alice_policy"].observation_space
            bob_bc_post_batch = self.policy_map["bob_policy"].relable_demonstration(alice_post_batch, alice_observation_space)
            policy_collector_group.policy_collectors["bob_policy"].add_postprocessed_batch_for_training(
                bob_bc_post_batch, self.policy_map["bob_policy"].view_requirements
            )
        if policy_collector_group:
            env_steps = self.episode_steps[episode_id]
            policy_collector_group.env_steps += env_steps
            agent_steps = self.agent_steps[episode_id]
            policy_collector_group.agent_steps += agent_steps

        if is_done:
            del self.episode_steps[episode_id]
            del self.agent_steps[episode_id]
            del self.episodes[episode_id]

            if policy_collector_group:
                self.policy_collector_groups[env_id].append(policy_collector_group)
        else:
            self.episode_steps[episode_id] = self.agent_steps[episode_id] = 0

        # Build a MultiAgentBatch from the episode and return.
        if build and build_next_batch:
            return self._build_multi_agent_batch(episode)
    