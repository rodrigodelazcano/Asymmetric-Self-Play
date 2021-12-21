import attr
import numpy as np

from typing import Tuple
from .base import (
    BaseEnv,
    BaseEnvParameters,
    BaseEnvConstants
)
from .robot_env import build_nested_attr

from .simulation import (
    AliceSim,
    AliceSimParameters,
)

## ALICE ENV PARAMETERS
@attr.s(auto_attribs=True)
class AliceEnvParameters(BaseEnvParameters):
    simulation_params: AliceSimParameters = build_nested_attr(
        AliceSimParameters
    )

## ALICE BUILD INTERFACE
class AliceEnv(
    BaseEnv[AliceEnvParameters, BaseEnvConstants, AliceSim]
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

        # valid/invalid goal setting
        self.valid_goal = True

    def _sample_object_colors(self, num_groups: int):
        return self._random_state.permutation(self.OBJECT_COLORS)[:num_groups]

    def _reset(self):
        super()._reset()

        # Save initial object pose
        self.initial_object_pos = self.mujoco_simulation.get_object_pos(pad=False).copy()
        self.initial_object_quat = self.mujoco_simulation.get_object_quat(pad=False).copy()

        self.valid_goal = True
    
    def check_objects_moved_from_init_pose(self) -> bool:
        current_object_pos = self.mujoco_simulation.get_object_pos(pad=False).copy()
        current_object_quat = self.mujoco_simulation.get_object_quat(pad=False).copy()

        if (
            np.allclose(current_object_pos, self.initial_object_pos, atol=1e-05) and 
            np.allclose(current_object_quat, self.initial_object_quat, atol=1e-05)
            ):

            return False
        
        return True
            
    
    def _get_simulation_info(self) -> dict:
        simulation_info = super()._get_simulation_info()
        
        object_positions = self.mujoco_simulation.get_object_pos()[
            : self.mujoco_simulation.num_objects
        ]
        # check if the initial pose of the object is different from the final goal pose. Return True
        # objects in different pose 
        objects_dif_pos_inital = self.check_objects_moved_from_init_pose()
        objects_in_placement_area = self.mujoco_simulation.check_objects_in_placement_area(
            object_positions
            )

        simulation_info["objects_in_placement_area"] = objects_in_placement_area
        simulation_info["objects_moved"] = objects_dif_pos_inital

        return simulation_info

    def _get_simulation_reward_with_done(self, info: dict) -> Tuple[float, bool]:

        reward, done = super()._get_simulation_reward_with_done(info)

        if (
            self.t >= self.constants.max_goal_setting_steps_per_object * 
            self.mujoco_simulation.simulation_params.num_objects
        ):
            done = True

        # If there is a gripper to table contact, apply table_collision penalty
        table_penalty = self.parameters.simulation_params.penalty.get(
            "table_collision", 0.0
        )

        if self.mujoco_simulation.get_gripper_table_contact():
            reward -= table_penalty

        # Add a large negative penalty for "breaking" the wrist camera by hitting it
        # with the table or another object.
        if info.get("wrist_cam_contacts", {}).get("any", False):
            reward -= self.parameters.simulation_params.penalty.get(
                "wrist_collision", 0.0
            )

        if done:
            if "valid_goal" in info and info["valid_goal"]:
                reward += 1.0
            if "objects_moved" in info and not info["objects_moved"]:
                reward -= self.parameters.simulation_params.penalty.get(
                    "objects_did_not_move", 0.0
                )
                self.valid_goal = False
            elif "objects_off_table" in info and info["objects_off_table"].any():
                # If any object is off the table, we will end the episode
                # Add a penalty to letting an object go off the table
                reward -= self.parameters.simulation_params.penalty.get(
                    "objects_off_table", 0.0
                )
                self.valid_goal = False
            elif "objects_in_placement_area" in info and not info["objects_in_placement_area"].all():
                # If any object is off the table, we will end the episode
                done = True
                # Add a penalty to letting an object go out of the placement area
                reward -= self.parameters.simulation_params.penalty.get(
                    "object_out_placement_area", 0.0
                )
            
        if any(self.observe()["safety_stop"]):
            reward -= self.parameters.simulation_params.penalty.get("safety_stop", 0.0)
    
        return reward, done
    
    def get_simulation_info_reward_with_done(self):
        info, env_reward, done = super().get_simulation_info_reward_with_done()

        # Update info with valid/invalide goal setting
        # and latest object pose
        info["valid_goal"] = self.valid_goal
        info["last_object_pos"] = self.mujoco_simulation.get_object_pos(pad=False).copy()
        info["last_obj_quat"] = self.mujoco_simulation.get_object_quat(pad=False).copy()
        return info, env_reward, done

        
make_env = AliceEnv.build