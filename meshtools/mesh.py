import numpy as np

EPSILON = 1e-10


class Mesh:
    def __init__(self, vertices, faces, normals, uvs, materials, group_names,
                 object_names, center=True):
        self.vertices = vertices
        self.faces = faces
        self.normals = normals
        self.uvs = uvs[:, :2]
        self.materials = materials
        self.group_names = group_names
        self.object_names = object_names

        max = self.vertices.max(axis=0)
        min = self.vertices.min(axis=0)
        center_point = (max + min) / 2

        if center:
            self.vertices -= center_point[None, :]

    def get_faces(self, filter=None):
        if filter is None:
            return self.faces
        faces = []
        for face in self.faces:
            for k, v in filter.items():
                if face[k] == v:
                    faces.append(face)
        return faces

    def expand_tangents(self, filter=None):
        tangents = []
        bitangents = []
        for face in self.get_faces(filter):
            face_vertex_indices = [v for v in face['vertices']]
            face_uv_indices = [v for v in face['uvs']]
            if None not in face_uv_indices:
                face_vertices = [self.vertices[i - 1, :]
                                 for i in face_vertex_indices]
                face_uvs = [self.uvs[i - 1, :] * 100
                            for i in face_uv_indices]
                delta_pos1 = face_vertices[1] - face_vertices[0]
                delta_pos2 = face_vertices[2] - face_vertices[0]
                delta_uv1 = face_uvs[1] - face_uvs[0]
                delta_uv2 = face_uvs[2] - face_uvs[0]
                denom = (delta_uv1[0] * delta_uv2[1]
                           - delta_uv1[1] * delta_uv2[0])
                r = 1.0 / (denom if denom > 0 else EPSILON)
                tangent = r * (delta_pos1 * delta_uv2[1]
                               - delta_pos2 * delta_uv1[1])
                bitangent = r * (delta_pos2 * delta_uv1[0]
                                 - delta_pos1 * delta_uv2[0])
            else:
                tangent = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                bitangent = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            tangents.extend([tangent] * 3)
            bitangents.extend([bitangent] * 3)

        return np.array(tangents), np.array(bitangents)

    def expand_face_vertices(self, filter=None):
        out_vertices = []
        for face in self.get_faces(filter):
            face_vertex_indices = [v for v in face['vertices']]
            face_vertices = [self.vertices[i - 1, :]
                             for i in face_vertex_indices]
            out_vertices.extend(face_vertices)
        return np.array(out_vertices)

    def expand_face_uvs(self, filter=None):
        out_uvs = []
        for face in self.get_faces(filter):
            face_uv_indices = [v for v in face['uvs']]
            if None not in face_uv_indices:
                face_uvs = [self.uvs[i - 1, :]
                            for i in face_uv_indices]
            else:
                # Add placeholder UVs if not available.
                zero = np.array([0.0, 0.0], dtype=np.float32)
                face_uvs = [zero] * 3
            out_uvs.extend(face_uvs)
        return np.array(out_uvs)

    def expand_face_normals(self, filter=None):
        out_normals = []
        for face in self.get_faces(filter):
            face_vertex_indices = [v for v in face['normals']]
            face_normals = [self.normals[i - 1, :]
                            for i in face_vertex_indices]
            out_normals.extend(face_normals)
        return np.array(out_normals)

    def bounding_size(self):
        max = self.vertices.max(axis=0)
        min = self.vertices.min(axis=0)
        return np.max(max - min)

    def resize(self, size):
        self.vertices *= size / self.bounding_size()

    def num_segments(self, segment_type='material'):
        if segment_type == 'material':
            return len(self.materials)
        elif segment_type == 'object':
            return len(self.object_names)
        else:
            return len(self.group_names)

    def build_mtl(self):
        str = ""
        for material in self.materials.values():
            str += material.build_mtl()
        return str


class Material:
    def __init__(self, name, index):
        self.name = name
        self.index = index
        self.specular_exponent = 2
        self.specular_color = (0.0, 0.0, 0.0)
        self.diffuse_color = (1.0, 1.0, 1.0)
        self.ambient_color = (0.0, 0.0, 0.0)
        self.emmissive_color = (0.0, 0.0, 0.0)

    def build_mtl(self):
        str = """
newmtl {}
Ns {}
Ks {} {} {}
Ka {} {} {}
Kd {} {} {}
Ke {} {} {}
illum 2
        """.format(self.name,
                   self.specular_exponent,
                   self.specular_color[0], self.specular_color[1],
                   self.specular_color[2],
                   self.ambient_color[0], self.ambient_color[1],
                   self.ambient_color[2],
                   self.diffuse_color[0], self.diffuse_color[1],
                   self.diffuse_color[2],
                   self.emmissive_color[0], self.emmissive_color[1],
                   self.emmissive_color[2])
        return str
