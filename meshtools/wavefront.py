import logging
import re
from collections import OrderedDict

import numpy as np

from meshtools.mesh import Material, Mesh

OBJ_COMMENT_MARKER = '#'
OBJ_VERTEX_MARKER = 'v'
OBJ_NORMAL_MARKER = 'vn'
OBJ_UV_MARKER = 'vt'
OBJ_FACE_MARKER = 'f'
OBJ_MTL_LIB_MARKER = 'mtllib'
OBJ_MTL_USE_MARKER = 'usemtl'
OBJ_GROUP_NAME_MARKER = 'g'
OBJ_OBJECT_NAME_MARKER = 'o'

MTL_COMMENT_MARKER = '#'
MTL_NEWMTL_MARKER = 'newmtl'
MTL_SPECULAR_EXPONENT_MARKER = 'Ns'
MTL_SPECULAR_COLOR_MARKER = 'Ks'
MTL_DIFFUSE_COLOR_MARKER = 'Kd'
MTL_AMBIENT_COLOR_MARKER = 'Ka'
MTL_EMMISSIVE_COLOR_MARKER = 'Ke'

logger = logging.getLogger(__name__)


def __parse_face(parts, material_id, group_id, object_id):
    face_vertices = []
    face_normals = []
    face_uvs = []
    for i in [1, 2, 3]:
        face_vertex_def = parts[i]
        split = face_vertex_def.split('/')

        vertex_idx = int(split[0])
        uv_idx = int(split[1]) if (len(split) > 1 and
                                   len(split[1]) > 0) else None
        normal_idx = int(split[2]) if (len(split) > 2 and
                                       len(split[2]) > 0) else None

        face_vertices.append(vertex_idx)
        face_normals.append(normal_idx)
        face_uvs.append(uv_idx)

    return {
        'vertices': face_vertices,
        'normals': face_normals,
        'uvs': face_uvs,
        'material': material_id,
        'group': group_id,
        'object': object_id,
    }


def read_obj_file(path):
    vertices = []
    faces = []
    normals = []
    uvs = []

    material_ids = OrderedDict([])
    material_counter = -1
    current_material_id = -1

    group_ids = {}
    group_counter = -1
    current_group_id = -1

    object_ids = {}
    object_counter = -1
    current_object_id = -1

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            # Ignore comments and whitespace.
            if len(line) < 3 or line[0] == OBJ_COMMENT_MARKER:
                continue

            parts = re.split(r'\s+', line)
            if parts[0] == OBJ_VERTEX_MARKER:
                vertex = [float(v) for v in parts[1:]]
                vertices.append(vertex)
            elif parts[0] == OBJ_NORMAL_MARKER:
                normal = [float(n) for n in parts[1:]]
                normals.append(normal)
            elif parts[0] == OBJ_UV_MARKER:
                uv = [float(u) for u in parts[1:]]
                uvs.append(uv)
            elif parts[0] == OBJ_FACE_MARKER:
                faces.append(__parse_face(parts,
                                          current_material_id,
                                          current_group_id,
                                          current_object_id))
            elif parts[0] == OBJ_MTL_USE_MARKER:
                material_name = parts[1]
                if material_name not in material_ids:
                    material_counter += 1
                    material_ids[material_name] = material_counter
                current_material_id = material_ids[material_name]
            elif parts[0] == OBJ_GROUP_NAME_MARKER:
                group_name = parts[1]
                if group_name not in group_ids:
                    group_counter += 1
                    group_ids[group_name] = group_counter
                current_group_id = group_ids[group_name]
            elif parts[0] == OBJ_OBJECT_NAME_MARKER:
                object_name = parts[1]
                if object_name not in object_ids:
                    object_counter += 1
                    object_ids[object_name] = object_counter
                current_object_id = object_ids[object_name]

    materials = OrderedDict([])
    for name in material_ids:
        materials[name] = Material(name, material_ids[name])

    return Mesh(np.array(vertices, dtype=np.float32),
                faces,
                np.array(normals, dtype=np.float32),
                np.array(uvs, dtype=np.float32),
                materials,
                group_ids,
                object_ids)


def read_mtl_file(path, model):
    materials = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            # Ignore comments and whitespace.
            if len(line) < 3 or line[0] == OBJ_COMMENT_MARKER:
                continue
            parts = re.split(r'\s+', line)

            if parts[0] == MTL_NEWMTL_MARKER:
                material_name = parts[1]
                if material_name not in model.materials:
                    raise ValueError(
                        'Material name {} not present in model'.format(
                            material_name))
                materials[material_name] = Material(material_name,
                                                    len(materials))
                current_material = materials[material_name]
            elif parts[0] == MTL_SPECULAR_EXPONENT_MARKER:
                current_material.specular_exponent = float(parts[1])
            elif parts[0] == MTL_SPECULAR_COLOR_MARKER:
                components = [float(c) for c in parts[1:]]
                current_material.specular_color = components
            elif parts[0] == MTL_DIFFUSE_COLOR_MARKER:
                components = [float(c) for c in parts[1:]]
                current_material.diffuse_color = components
            elif parts[0] == MTL_AMBIENT_COLOR_MARKER:
                components = [float(c) for c in parts[1:]]
                current_material.ambient_color = components
            elif parts[0] == MTL_EMMISSIVE_COLOR_MARKER:
                components = [float(c) for c in parts[1:]]
                current_material.emmissive_color = components

    return materials
