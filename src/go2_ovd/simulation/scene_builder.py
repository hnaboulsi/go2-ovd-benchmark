from __future__ import annotations

import math
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np

from go2_ovd.config import repository_root


def mesh_bounds(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    with Path(path).open("r", encoding="utf-8", errors="strict") as stream:
        for line in stream:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
    if not vertices:
        raise ValueError(f"mesh contains no vertices: {path}")
    array = np.asarray(vertices, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"mesh contains non-finite vertices: {path}")
    return array.min(axis=0), array.max(axis=0)


def _copy_robot(controller_root: Path, output_dir: Path, camera: dict) -> None:
    source = controller_root / "models" / "MJCF" / "go2"
    shutil.copytree(source / "assets", output_dir / "assets", dirs_exist_ok=True)
    tree = ET.parse(source / "go2.xml")
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError("upstream Go2 MJCF has no compiler element")
    compiler.set("texturedir", "assets")
    base = root.find(".//body[@name='base_link']")
    if base is None:
        raise ValueError("upstream Go2 MJCF has no base_link body")
    pitch = math.radians(float(camera["downward_pitch_deg"]))
    y_axis = (math.sin(pitch), 0.0, math.cos(pitch))
    position = " ".join(f"{float(value):.9g}" for value in camera["body_position_m"])
    xyaxes = f"0 -1 0 {y_axis[0]:.9g} 0 {y_axis[2]:.9g}"
    ET.SubElement(
        base,
        "camera",
        {
            "name": str(camera["name"]),
            "pos": position,
            "xyaxes": xyaxes,
            "fovy": f"{float(camera['vertical_fov_deg']):.9g}",
        },
    )
    tree.write(output_dir / "go2_with_camera.xml", encoding="utf-8", xml_declaration=True)


def _object_asset(root: Path, object_name: str) -> tuple[Path, Path, np.ndarray, np.ndarray]:
    ycb_id = {"baseball": "055_baseball", "softball": "054_softball"}[object_name]
    source = root / ".cache" / "assets" / "ycb" / ycb_id / "google_16k"
    mesh = source / "textured.obj"
    texture = source / "texture_map.png"
    if not mesh.exists() or not texture.exists():
        raise FileNotFoundError(f"missing downloaded YCB asset for {object_name}; run bootstrap first")
    minimum, maximum = mesh_bounds(mesh)
    return mesh, texture, minimum, maximum


def build_scene(
    output_dir: str | Path,
    controller_root: str | Path,
    camera: dict,
    condition: str,
    target_center_xy: tuple[float, float],
    view_yaw_rad: float,
) -> dict:
    root = repository_root()
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    _copy_robot(Path(controller_root), target, camera)
    ycb_output = target / "assets" / "ycb"
    ycb_output.mkdir(parents=True, exist_ok=True)

    included = {
        "baseball": condition in {"target", "target_distractor", "target_distractor_occluded"},
        "softball": condition in {"target_distractor", "distractor_only", "target_distractor_occluded"},
    }
    offsets = {
        "baseball": -0.12 if included["baseball"] and included["softball"] else 0.0,
        "softball": 0.12 if included["baseball"] and included["softball"] else 0.0,
    }
    lateral = np.array([-math.sin(view_yaw_rad), math.cos(view_yaw_rad)], dtype=float)
    center = np.asarray(target_center_xy, dtype=float)

    object_info: dict[str, dict] = {}
    for name, present in included.items():
        if not present:
            continue
        mesh, texture, minimum, maximum = _object_asset(root, name)
        object_dir = ycb_output / name
        object_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mesh, object_dir / "textured.obj")
        shutil.copy2(mesh.with_name("textured.mtl"), object_dir / "textured.mtl")
        shutil.copy2(texture, object_dir / "texture_map.png")
        position_xy = center + offsets[name] * lateral
        object_info[name] = {
            "minimum": minimum,
            "maximum": maximum,
            "extent": maximum - minimum,
            "world_xy": position_xy,
            "geom_offset": np.array([-(minimum[0] + maximum[0]) / 2, -(minimum[1] + maximum[1]) / 2, -minimum[2]]),
        }

    model = ET.Element("mujoco", {"model": "go2 ovd perception scene"})
    ET.SubElement(model, "include", {"file": "go2_with_camera.xml"})
    visual = ET.SubElement(model, "visual")
    ET.SubElement(visual, "headlight", {"diffuse": "0.6 0.6 0.6", "ambient": "0.35 0.35 0.35", "specular": "0 0 0"})
    asset = ET.SubElement(model, "asset")
    ET.SubElement(asset, "texture", {"type": "2d", "name": "groundplane", "builtin": "checker", "rgb1": "0.38 0.38 0.38", "rgb2": "0.24 0.24 0.24", "width": "256", "height": "256"})
    ET.SubElement(asset, "material", {"name": "groundplane", "texture": "groundplane", "texuniform": "true", "texrepeat": "8 8", "reflectance": "0"})
    for name in object_info:
        ET.SubElement(asset, "mesh", {"name": f"{name}_mesh", "file": f"ycb/{name}/textured.obj"})
        ET.SubElement(asset, "texture", {"type": "2d", "name": f"{name}_texture", "file": f"ycb/{name}/texture_map.png"})
        ET.SubElement(asset, "material", {"name": f"{name}_material", "texture": f"{name}_texture", "specular": "0.05", "shininess": "0.1"})
    world = ET.SubElement(model, "worldbody")
    ET.SubElement(world, "light", {"name": "key_light", "pos": "-1 -1 2.5", "dir": "0.3 0.2 -1", "directional": "true", "diffuse": "0.9 0.9 0.9"})
    ET.SubElement(world, "geom", {"name": "floor", "size": "0 0 0.05", "type": "plane", "material": "groundplane", "friction": "0.8 0.02 0.01"})
    for name, info in object_info.items():
        body = ET.SubElement(world, "body", {"name": f"{name}_0", "pos": f"{info['world_xy'][0]:.9g} {info['world_xy'][1]:.9g} 0"})
        offset = info["geom_offset"]
        ET.SubElement(body, "geom", {
            "name": f"{name}_geom",
            "type": "mesh",
            "mesh": f"{name}_mesh",
            "material": f"{name}_material",
            "pos": f"{offset[0]:.9g} {offset[1]:.9g} {offset[2]:.9g}",
            "contype": "0",
            "conaffinity": "0",
        })
    if condition == "target_distractor_occluded":
        ET.SubElement(world, "geom", {
            "name": "target_occluder",
            "type": "box",
            "pos": f"{center[0]:.9g} {center[1]:.9g} 0.05",
            "size": "0.01 0.025 0.055",
            "rgba": "0.35 0.35 0.35 1",
            "group": "4",
            "contype": "0",
            "conaffinity": "0",
        })
    scene_path = target / "scene.xml"
    ET.ElementTree(model).write(scene_path, encoding="utf-8", xml_declaration=True)
    return {
        "scene_path": scene_path,
        "objects": {
            name: {
                "extent_m": info["extent"].tolist(),
                "world_xy": info["world_xy"].tolist(),
            }
            for name, info in object_info.items()
        },
    }

