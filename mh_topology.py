import numpy as np
import openmesh as om
from pathlib import Path
from utils.mesh import get_obj_face, get_obj_vertices


def mh_to_flame(mh_path: Path, mapping, append_filename: str = "mapped"):
    
    print(f"Converting MH obj: {mh_path} to FLAME")
    
    verts_mh = get_obj_vertices(mh_path)
    verts_mh = np.asarray(verts_mh)

    # Adding eyeball vertices to mh
    verts_mh = np.concatenate((verts_mh, left_eyeball, right_eyeball), axis=0)
    assert verts_mh.shape == (5023, 3)

    # Reordering mh
    verts_mh_flame = verts_mh[mapping, :]

    # Replacing eyes with the closest vertex on the face
    verts_mh_flame[left_eyeball_rows, :] = verts_mh_flame[left_id, :]
    verts_mh_flame[right_eyeball_rows, :] = verts_mh_flame[right_id, :]

    npy_out_path = (mh_path.parent / "flame").with_suffix(".npy")
    np.save(npy_out_path, verts_mh_flame)

    mesh = om.TriMesh()

    v = np.array([mesh.add_vertex(v) for v in verts_mh_flame])
    f = [mesh.add_face(list(v[t - 1])) for t in faces_flame]

    mesh_out_name = f"{mh_path.stem}_{append_filename}"
    mesh_out_path = (mh_path.parent / mesh_out_name).with_suffix(".obj")
    om.write_mesh(str(mesh_out_path), mesh)


def create_mappings(from_mesh, to_mesh):

    # Finding closest mh point to each flame point
    mapped = [np.linalg.norm(from_mesh - i, axis=1).argmin() for i in to_mesh]
    assert len(mapped) == len(np.unique(mapped))
    
    return mapped


if __name__ == "__main__":

    data_path = Path("/mnt/disks/data/datasets")
    flame_template = Path("/home/andrew/templates/head_template.obj")
    mh_template = Path("/home/andrew/templates/head_template_noeyes_MH_UV.obj")

    verts_flame = np.asarray(get_obj_vertices(flame_template))
    faces_flame = np.asarray(get_obj_face(flame_template), dtype=int)

    verts_mh = np.asarray(get_obj_vertices(mh_template))

    ##### Eyes #####
    # Finding a vertex on the face that is close to each eyeball
    face_rows = range(3931)
    left_eyeball_rows = range(3931, 4477)
    right_eyeball_rows = range(4477, 5023)

    face = verts_flame[face_rows, :]
    left_eyeball = verts_flame[left_eyeball_rows, :]
    right_eyeball = verts_flame[right_eyeball_rows, :]
    assert left_eyeball.shape == right_eyeball.shape

    left_id = np.linalg.norm(face - left_eyeball.mean(axis=0), axis=1).argmin()
    right_id = np.linalg.norm(face - right_eyeball.mean(axis=0), axis=1).argmin()
    ################

    # Adding eyeball vertices to mh
    verts_mh = np.concatenate((verts_mh, left_eyeball, right_eyeball), axis=0)
    print(verts_mh.shape)

    mh2flame = create_mappings(from_mesh=verts_mh, to_mesh=verts_flame)
    flame2mh = create_mappings(from_mesh=verts_flame, to_mesh=verts_mh)

    print(np.array(mh2flame)[face_rows].min(), np.array(mh2flame)[face_rows].max())

    np.save("mh2flame_mapping.npy", np.array(mh2flame)[face_rows])
    np.save("flame2mh_mapping.npy", np.array(flame2mh)[face_rows])

    append_with = "flame"
    mh_to_flame(mh_template, mh2flame, append_with)

    for obj_file in data_path.rglob("*.obj"):
        if "final" in obj_file.name and not obj_file.stem.endswith(append_with):
            try:
                mh_to_flame(obj_file, mh2flame, append_with)
            except AssertionError as e:
                print(f"Obj file ({obj_file}) has wrong shape: {e}")