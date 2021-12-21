from typing import Dict, Optional, Tuple, Union

import attr
import numpy as np
import random

from robogym.envs.rearrange.common.base import (
    RearrangeEnv,
    RearrangeEnvConstants,
    RearrangeEnvParameters,
)

from robogym.envs.rearrange.common.utils import (
    place_objects_in_grid,
    place_objects_with_no_constraint,
)

from robogym.envs.rearrange.simulation.blocks import (
    BlockRearrangeSim,
    BlockRearrangeSimParameters,
)
from robogym.envs.rearrange.simulation.base import RearrangeSimulationInterface
from robogym.robot_env import build_nested_attr


@attr.s(auto_attribs=True)
class BobEnvParameters(RearrangeEnvParameters):
    simulation_params: BlockRearrangeSimParameters = build_nested_attr(
        BlockRearrangeSimParameters
    )

from robogym.envs.rearrange.goals.object_state import GoalArgs, ObjectStateGoal

class BobEnv(
    RearrangeEnv[BobEnvParameters, RearrangeEnvConstants, BlockRearrangeSim]
):
    OBJECT_COLORS = (
        (1, 0, 0, 1),
        (0, 1, 0, 1),
        (0, 0, 1, 1),
        (1, 1, 0, 1),
        (0, 1, 1, 1),
        (1, 0, 1, 1),
        (0, 0, 0, 1),
        (1, 1, 1, 1),
    )
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        init_quats = np.array([[1, 0, 0, 0]] * self.mujoco_simulation.num_objects)
        init_pos, _= self._generate_object_placements()

        assert init_pos.shape == (self.mujoco_simulation.num_objects, 3)
        assert init_quats.shape == (self.mujoco_simulation.num_objects, 4)
        self.set_initial_state_and_goal_pose(init_pos=init_pos, init_quat=init_quats, goal_pos=, goal_quat=)
    def _sample_object_colors(self, num_groups: int):
        return self._random_state.permutation(self.OBJECT_COLORS)[:num_groups]

    def set_initial_state_and_goal_pose(
        self, init_pos: np.ndarray, init_quat: np.ndarray,
        goal_pos: np.ndarray, goal_quat: np.ndarray
    ) -> None:
        
        ### assert test here ###

        self.init_pos = init_pos
        self.init_quat = init_quat
        self.goal_pos = goal_pos
        self.goal_quat = goal_quat

    def _reset(self):
        super()._reset()
        self.mujoco_simulation.set_object_pos(self.init_pos)
        self.mujoco_simulation.set_object_quat(self.init_quat)

        self.goal_generation.set_goal_pose(self.goal_pos, self.goal_quat)

    # we don't need to check if objects off table,
    # only if goals are achieved after T steps

    # Agent is done after T steps

    # in _reset() set initial object pose from alice and final goal pose

    # what about objects color? import for camera observations

    # check when env is done and confirm reward structure

    # Create a GoalArg class to set desired goals

    @classmethod
    def build_goal_generation(cls, constants, mujoco_simulation):
        return SetStateGoal(
            mujoco_simulation,
            args=constants.goal_args,
        )

class SetStateGoal(ObjectStateGoal):
    def __init__(
        self,
        mujoco_simulation: RearrangeSimulationInterface,
        args: Union[Dict, GoalArgs],
        init_quats: Optional[np.ndarray] = None,
        init_pos: Optional[np.ndarray] = None,
        starting_seed: Optional[int] = None,
    ):
        """
        Set a user defined goal state.

        :param relative_placements: the relative position of each object, relative to the placement area.
            Each dimension is a ratio between [0, 1].
        :param init_quats: the desired quat of each object.
        """
        super().__init__(mujoco_simulation, args)

        self._last_seed = (
            starting_seed
            if starting_seed is not None
            else random.randint(0, 2 ** 32 - 1)
        )
        self.random_state = np.random.RandomState(self._last_seed)
        
        if init_quats is None:
            init_quats = np.array([[1, 0, 0, 0]] * self.mujoco_simulation.num_objects)
        
        if init_pos is None:
            init_pos, is_valid = self._generate_object_placements()
            assert is_valid, "Initial goal setting not valid"
        assert init_pos.shape == (self.mujoco_simulation.num_objects, 3)
        assert init_quats.shape == (self.mujoco_simulation.num_objects, 4)

        self.goal_pos = init_pos
        self.goal_quats = init_quats
        
    
    def _generate_object_placements(self) -> Tuple[np.ndarray, bool]:
        """
        Generate placement for each object. Return object placement and a
        boolean indicating whether the placement is valid.
        """
        placement, is_valid = place_objects_in_grid(
            self.mujoco_simulation.get_object_bounding_boxes(),
            self.mujoco_simulation.get_table_dimensions(),
            self.mujoco_simulation.get_placement_area(),
            random_state=self.random_state,
            max_num_trials=self.mujoco_simulation.max_placement_retry,
        )

        if not is_valid:
            # Fall back to random placement, which works better for envs with more irregular
            # objects (e.g. ycb-8 with no mesh normalization).
            return place_objects_with_no_constraint(
                self.mujoco_simulation.get_object_bounding_boxes(),
                self.mujoco_simulation.get_table_dimensions(),
                self.mujoco_simulation.get_placement_area(),
                max_placement_trial_count=self.mujoco_simulation.max_placement_retry,
                max_placement_trial_count_per_object=self.mujoco_simulation.max_placement_retry_per_object,
                random_state=self.random_state,
            )
        else:
            return placement, is_valid

    
    def _update_simulation_for_next_goal(self, random_state: np.random.RandomState) -> Tuple[bool, Dict[str, np.ndarray]]:
        
        quats = np.array(self.goal_quats)
        pos = np.array(self.goal_pos)
        self.mujoco_simulation.set_target_quat(quats)
        self.mujoco_simulation.set_target_pos(pos)
        self.mujoco_simulation.forward()

        if self.args.stabilize_goal:
            self._stablize_goal_objects()

        goal = {
            "obj_pos": self.mujoco_simulation.get_target_pos().copy(),
            "obj_rot": self.mujoco_simulation.get_target_rot().copy(),
        }
        return True, goal
        
    def set_goal_pose(
        self, goal_pos: np.ndarray, goal_quat: np.ndarray
    ) -> None:
        self.goal_pos = goal_pos
        self.goal_quats = goal_quat

make_env = BobEnv.build