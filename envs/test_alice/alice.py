import attr
from typing import Any, Dict, Generic, List, Optional, Tuple, Type, TypeVar, Union
import numpy as np
import logging

from robogym.mujoco.mujoco_xml import ASSETS_DIR, MujocoXML

from robogym.wrappers.util import (
    BinSpacing,
)

from base import (
    BaseEnv,
    BaseEnvParameters,
)
from robogym.robot.robot_interface import RobotControlParameters

from robot_env import ObservationMapValue as omv
from robot_env import (
    RobotEnvConstants,
    build_nested_attr,
    get_generic_param_type,
)

from simulation import (
    AliceSimulationInterface,
    AliceSimParameters,
)
from mujoco_py import MjSim

from robogym.utils.rotation import mat2quat, quat2mat, quat_conjugate, uniform_quat

## Utils functions

def rotate_bounding_box(
    bounding_box: np.ndarray, quat: np.ndarray
) -> Tuple[float, float]:
    """ Rotates a bounding box by applying the quaternion and then re-computing the tightest
    possible fit of an *axis-aligned* bounding box.
    """
    pos, size = bounding_box

    # Compute 8 corners of bounding box.
    signs = np.array([[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]])
    corners = pos + signs * size
    assert corners.shape == (8, 3)

    # Rotate corners.
    mat = quat2mat(quat)
    rotated_corners = corners @ mat

    # Re-compute bounding-box.
    min_xyz = np.min(rotated_corners, axis=0)
    max_xyz = np.max(rotated_corners, axis=0)
    size = (max_xyz - min_xyz) / 2.0
    assert np.all(size >= 0.0)
    pos = min_xyz + size

    return pos, size

def geom_ids_of_body(sim: MjSim, body_name: str) -> List[int]:
    object_id = sim.model.body_name2id(body_name)
    object_geomadr = sim.model.body_geomadr[object_id]
    object_geomnum = sim.model.body_geomnum[object_id]
    return list(range(object_geomadr, object_geomadr + object_geomnum))

def get_block_bounding_box(sim, object_name) -> Tuple[float, float]:
    """ Returns the bounding box of a block body. If the block is rotated in the world frame,
    the rotation is applied and the tightest axis-aligned bounding box is returned.
    """
    geom_ids = geom_ids_of_body(sim, object_name)
    assert len(geom_ids) == 1, f"More than 1 geoms in {object_name}."
    geom_id = geom_ids[0]
    size = sim.model.geom_size[geom_id]
    pos = sim.model.geom_pos[geom_id]

    quat = quat_conjugate(mat2quat(sim.data.get_body_xmat(object_name)))
    pos, size = rotate_bounding_box((pos, size), quat)
    return pos, size

def make_openai_block(name: str, object_size: np.ndarray) -> MujocoXML:
    """ Creates a block with OPENAI letters on it faces.

    :param name: The name of the block
    :param object_size: The size of the block (3-dimensional). This is half-size as per Mujoco
        convention
    """
    default_object_size = 0.0285
    default_letter_offset = 0.0009

    # scale face meshes properly
    scale = object_size / default_object_size
    letter_offset = default_letter_offset * scale

    def to_str(x: np.ndarray):
        return " ".join(map(str, x.tolist()))

    face_pos = {
        "top": {
            "body": to_str(np.array([0, 0, object_size[2]])),
            "geom": to_str(np.array([0, 0, -letter_offset[2]])),
        },
        "bottom": {
            "body": to_str(np.array([0, 0, -object_size[2]])),
            "geom": to_str(np.array([0, 0, letter_offset[2]])),
        },
        "back": {
            "body": to_str(np.array([0, object_size[1], 0])),
            "geom": to_str(np.array([0, -letter_offset[1], 0])),
        },
        "right": {
            "body": to_str(np.array([object_size[0], 0, 0])),
            "geom": to_str(np.array([-letter_offset[0], 0, 0])),
        },
        "front": {
            "body": to_str(np.array([0, -object_size[1], 0])),
            "geom": to_str(np.array([0, letter_offset[1], 0])),
        },
        "left": {
            "body": to_str(np.array([-object_size[0], 0, 0])),
            "geom": to_str(np.array([letter_offset[0], 0, 0])),
        },
    }
    face_euler = {
        "top": to_str(np.array([np.pi / 2, 0, np.pi / 2])),
        "bottom": to_str(np.array([np.pi / 2, 0, np.pi / 2])),
        "back": to_str(np.array([0, 0, np.pi / 2])),
        "right": to_str(np.array([0, 0, 0])),
        "front": to_str(np.array([0, 0, -np.pi / 2])),
        "left": to_str(np.array([0, 0, np.pi])),
    }

    def face_xml(_name: str, _face: str, _c: str):
        xml = f"""
        <body name="{_face}:{_name}" pos="{face_pos[_face]['body']}">
            <geom name="letter_{_c}:{_name}" mesh="{_name}{_c}" euler="{face_euler[_face]}"
             pos="{face_pos[_face]['geom']}" type="mesh" material="{_name}letter"
             conaffinity="0" contype="0" />
        </body>
        """
        return xml

    size_string = " ".join(map(str, list(object_size)))
    scale_string = " ".join(map(str, list(scale)))

    xml_source = f"""
    <mujoco>
        <asset>
            <material name="{name}letter" specular="1" shininess="0.3" rgba="1 1 1 1"/>
            <mesh name="{name}O" file="{ASSETS_DIR}/stls/openai_cube/O.stl"
             scale="{scale_string}" />
            <mesh name="{name}P" file="{ASSETS_DIR}/stls/openai_cube/P.stl"
             scale="{scale_string}" />
            <mesh name="{name}E" file="{ASSETS_DIR}/stls/openai_cube/E.stl"
             scale="{scale_string}" />
            <mesh name="{name}N" file="{ASSETS_DIR}/stls/openai_cube/N.stl"
             scale="{scale_string}" />
            <mesh name="{name}A" file="{ASSETS_DIR}/stls/openai_cube/A.stl"
             scale="{scale_string}" />
            <mesh name="{name}I" file="{ASSETS_DIR}/stls/openai_cube/I.stl"
             scale="{scale_string}" />
        </asset>
        <worldbody>
            <body name="{name}">
                <geom name="{name}" size="{size_string}" type="box"
                 rgba="0.0 0.0 0.0 0.0" material="block_mat"/>
                <joint name="{name}:joint" type="free"/>
                {face_xml(name, "top", "O")}
                {face_xml(name, "bottom", "P")}
                {face_xml(name, "back", "E")}
                {face_xml(name, "right", "N")}
                {face_xml(name, "front", "A")}
                {face_xml(name, "left", "I")}
            </body>
        </worldbody>
    </mujoco>
    """
    return MujocoXML.from_string(xml_source)

def make_block(name: str, object_size: np.ndarray) -> MujocoXML:
    """Creates a block.

    :param name: The name of the block
    :param object_size: The size of the block (3-dimensional). This is half-size as per Mujoco
        convention
    """
    xml_source = f"""
    <mujoco>
    <worldbody>
        <body name="{name}" pos="0.0 0.0 0.0">
        <geom type="box" rgba="0.0 0.0 0.0 0.0" material="block_mat"/>
        <joint name="{name}:joint" type="free"/>
        </body>
    </worldbody>
    </mujoco>
    """
    xml = MujocoXML.from_string(xml_source).set_objects_attr(
        tag="geom", size=object_size
    )

    return xml

def make_blocks(
    num_objects: int, block_size: Union[float, np.ndarray], appearance: str = "standard"
) -> List[Tuple[MujocoXML]]:
    if isinstance(
        block_size, (int, float, np.integer, np.floating)
    ) or block_size.shape == (1,):
        block_size = np.tile(block_size, 3)
    assert block_size.shape == (
        3,
    ), f"Bad block_size: {block_size}, expected float, np.ndarray(1,) or np.ndarray(3,)"

    if appearance == "standard":
        make_block_fn = make_block
    elif appearance == "openai":
        make_block_fn = make_openai_block

    xmls: List[Tuple[MujocoXML]] = []
    for i in range(num_objects):
        # add the block
        block_xml = make_block_fn(f"object{i}", block_size.copy())
        xmls.append((block_xml))
    return xmls

# Final Alice Simulation
class AliceSim(AliceSimulationInterface[AliceSimParameters]):
    """
    Move around a blocks of different colors on the table.
    """

    @classmethod
    def make_objects_xml(cls, xml, simulation_params: AliceSimParameters):
        return make_blocks(
            simulation_params.num_objects,
            simulation_params.object_size,
            appearance=simulation_params.block_appearance,
        )

    def _get_bounding_box(self, object_name):
        return get_block_bounding_box(self.mj_sim, object_name)

## ALICE ENV CONSTANTS

logger = logging.getLogger(__name__)

VISION_OBS = "vision_obs"
VISION_OBS_MOBILE = "vision_obs_mobile"
VISION_GOAL = "vision_goal"


@attr.s(auto_attribs=True)
class EncryptEnvConstants:
    enabled: bool = False

    use_dummy: bool = False

    input_shape: List[int] = [200, 200, 3]

    distortable_keys: List[str] = []

    encryption_prefix: str = "enc_"

    last_activation: str = "TANH"

    # Set hardcoded values for randomization parameters here.
    # This is useful for creating an evaluator with fixed parameters (i.e. not subject to ADR).
    param_values: Optional[Dict[str, Union[float, int]]] = None

    # Whether we should use different Encryption networks for each input key.
    use_separate_networks_per_key: bool = True


@attr.s(auto_attribs=True)
class VisionArgs:
    # The size of the rendered obs and goal images in pixel.
    image_size: int = 200

    # Whether rendering high-res images, only used in examine_vision.
    high_res_mujoco: bool = False

    # Names for static cameras.
    camera_names: List[str] = ["vision_cam_front"]

    # Camera used for mobile cameras. They are only used
    # for obs images.
    mobile_camera_names: List[str] = ["vision_cam_wrist"]

    def all_camera_names(self):
        """Returns all camera names specified by static and mobile camera lists."""
        return self.mobile_camera_names + self.camera_names
        
@attr.s(auto_attribs=True)
class AliceEnvConstants(RobotEnvConstants):
    mujoco_substeps: int = 40
    mujoco_timestep: float = 0.001

    # If this is set to true, new "masked_*" observation keys will be created with goal and
    # object states where objects outside the placement area will be zeroed out.
    mask_obs_outside_placement_area: bool = False

    # If use vision observation.
    vision: bool = False

    vision_args: VisionArgs = build_nested_attr(VisionArgs)

    # lambda range in exponential decay distribution for sampling duplicated object counts.
    sample_lam_low: float = 0.1
    sample_lam_high: float = 5.0

    encrypt_env_constants: EncryptEnvConstants = build_nested_attr(EncryptEnvConstants)

    # Action spacing function to be used by the DiscretizeActionWrapper
    action_spacing: BinSpacing = attr.ib(
        default=BinSpacing.LINEAR,
        validator=attr.validators.in_(BinSpacing),
        converter=BinSpacing,
    )

    def has_mobile_cameras_enabled(self):
        return self.vision and len(self.vision_args.mobile_camera_names) > 0

    # Fix initial state
    # for debugging purpose, this option starts episode from fixed initial state again and
    # again. Initial state is determined by the `starting_seed`.
    use_fixed_seed: bool = False

    #####################
    # Self play settings.

    stabilize_objects: bool = True

## ALICE ENV PARAMETERS

@attr.s(auto_attribs=True)
class AliceEnvParameters(BaseEnvParameters):
    simulation_params: AliceSimParameters = build_nested_attr(
        AliceSimParameters
    )

## ALICE BUILD INTERFACE

class AliceEnv(
    BaseEnv[AliceEnvParameters, AliceEnvConstants, AliceSim]
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

    def _sample_object_colors(self, num_groups: int):
        return self._random_state.permutation(self.OBJECT_COLORS)[:num_groups]


make_env = AliceEnv.build