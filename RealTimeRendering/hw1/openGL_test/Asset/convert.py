#!/usr/bin/env python3
"""
ply2obj.py - Convert an ASCII PLY file to a Wavefront OBJ file.

Usage:
    python ply2obj.py input.ply output.obj

The script expects an ASCII PLY with at least 'element vertex' (x, y, z)
and 'element face' (a list of vertex indices, 0‑based). All other
properties are ignored.
"""

import sys
import os

def parse_ply_header(f):
    """
    Read the PLY header and return a dict with:
        'vertex_count' : int
        'face_count'   : int
        'format'       : str (should be 'ascii')
    Also leaves the file pointer right after the 'end_header' line.
    """
    header = {}
    line = f.readline().strip()
    if line != 'ply':
        raise ValueError("Not a valid PLY file (missing 'ply' magic).")

    while True:
        line = f.readline().strip()
        if not line:
            raise ValueError("Unexpected EOF inside PLY header.")
        if line.startswith('comment'):
            continue
        parts = line.split()
        keyword = parts[0]
        if keyword == 'format':
            if parts[1] != 'ascii':
                raise ValueError(f"Unsupported PLY format: {parts[1]}. Only ASCII is supported.")
            header['format'] = parts[1]
        elif keyword == 'element':
            element_name = parts[1]
            count = int(parts[2])
            header[f'{element_name}_count'] = count
        elif keyword == 'end_header':
            break
        # Ignore 'property' lines – we assume the vertex property order is x,y,z
        # and face property is a list of vertex indices.

    if 'vertex_count' not in header:
        raise ValueError("Missing 'element vertex' in PLY header.")
    if 'face_count' not in header:
        raise ValueError("Missing 'element face' in PLY header.")
    return header

def convert_ply_to_obj(ply_path, obj_path):
    with open(ply_path, 'r') as f_in:
        header = parse_ply_header(f_in)
        vertex_count = header['vertex_count']
        face_count = header['face_count']

        # Read all vertices (x, y, z)
        vertices = []
        for _ in range(vertex_count):
            line = f_in.readline()
            while line.strip() == '' or line.strip().startswith('comment'):
                line = f_in.readline()
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Vertex line has fewer than 3 numbers: {line.strip()}")
            x, y, z = parts[0], parts[1], parts[2]
            vertices.append((x, y, z))

        # Read all faces
        faces = []
        for _ in range(face_count):
            line = f_in.readline()
            while line.strip() == '' or line.strip().startswith('comment'):
                line = f_in.readline()
            parts = line.split()
            if len(parts) < 4:   # at least one count + 3 indices for a triangle
                raise ValueError(f"Face line too short: {line.strip()}")
            count = int(parts[0])
            if count < 3:
                raise ValueError(f"Face has fewer than 3 vertices: {line.strip()}")
            indices = [int(idx) for idx in parts[1:1+count]]
            faces.append(indices)

    # Write OBJ
    with open(obj_path, 'w') as f_out:
        f_out.write("# Converted from PLY by ply2obj.py\n")
        # Use the input filename as the object name (without extension)
        obj_name = os.path.splitext(os.path.basename(ply_path))[0]
        f_out.write(f"o {obj_name}\n")

        # Vertices
        for v in vertices:
            f_out.write(f"v {v[0]} {v[1]} {v[2]}\n")

        # Faces (OBJ uses 1‑based indices)
        for face in faces:
            # Convert 0‑based to 1‑based
            f_out.write("f " + " ".join(str(idx + 1) for idx in face) + "\n")

    print(f"Successfully converted {ply_path} -> {obj_path}")
    print(f"  Vertices: {len(vertices)}")
    print(f"  Faces:    {len(faces)}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python ply2obj.py input.ply output.obj")
        sys.exit(1)

    ply_file = sys.argv[1]
    obj_file = sys.argv[2]

    if not os.path.isfile(ply_file):
        print(f"Error: input file '{ply_file}' not found.")
        sys.exit(1)

    try:
        convert_ply_to_obj(ply_file, obj_file)
    except Exception as e:
        print(f"Conversion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()