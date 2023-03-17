from pathlib import Path


def get_obj_vertices(obj_path: Path):
    """
    Quick way of extracting vertices from an obj file
    """

    with open(obj_path, "r") as fi:
        vertices = []
        for ln in fi:
            if ln.startswith("v "):
                vertices.append([float(n) for n in ln[2:].strip().split()])

    return vertices


def get_obj_face(obj_path: Path):
    """
    Quick way of extracting faces from an obj file. Only takes first number (vertex) not texture or normal indices
    """

    with open(obj_path, "r") as fi:
        vertices = []
        for ln in fi:
            if ln.startswith("f "):
                vertices.append([float(n.split("/")[0]) for n in ln[2:].strip().split()])

    return vertices