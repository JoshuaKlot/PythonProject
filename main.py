"""
Plot a 2D slice (constant z) of node-based data defined on an unstructured
tetrahedral gmsh mesh (.msh, version 2.2).

Usage (command line):
    python mesh_slice.py mesh1.msh data.txt --z 0.5
    python mesh_slice.py mesh1.msh data.txt --z 0.5 --xlim -1 1 --ylim -1 1 --nx 400 --ny 400

Or from PyCharm: edit the DEFAULTS block below and press run.

Pipeline:
  1. meshio reads the .msh  -> nodes + tetrahedra
  2. the data file is read   -> one value per node
  3. PyVista builds an UnstructuredGrid, attaches the data to the nodes
  4. a regular (x, y) grid at the requested z is sampled from the mesh
     (PyVista interpolates inside each tetrahedron)
  5. matplotlib imshow draws the result (points outside the mesh = NaN/blank)
"""
import argparse
import sys

import matplotlib.pyplot as plt
import meshio
import numpy as np
import pyvista as pv

# ----------------------------------------------------------------------------
# DEFAULTS (used when no command-line arguments are given, e.g. run in PyCharm)
# ----------------------------------------------------------------------------
DEFAULT_MESH = "Mesh slice project/Charging Mesh Results/mesh1.msh"
DEFAULT_DATA = "Mesh slice project/Charging Mesh Results/MeshShape_charge_density.msh"
DEFAULT_Z = 200.0
# ----------------------------------------------------------------------------


def load_mesh(path):
    """Read the .msh and return a PyVista UnstructuredGrid of its tetrahedra."""
    mesh = meshio.read(path)
    if "tetra" not in mesh.cells_dict:
        raise ValueError(
            f"No tetrahedra found in {path}. Cell types present: {list(mesh.cells_dict)}"
        )
    tets = mesh.cells_dict["tetra"]  # (n_tets, 4) indices into mesh.points

    # VTK cell array format: [4, a, b, c, d, 4, a, b, c, d, ...]
    n = len(tets)
    cells = np.hstack([np.full((n, 1), 4), tets]).ravel()
    celltypes = np.full(n, pv.CellType.TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, celltypes, mesh.points)

    # If the .msh itself carries point data, keep it (handy fallback).
    for name, arr in mesh.point_data.items():
        grid.point_data[name] = arr
    return grid


def get_node_tags(mesh_path, n_nodes):
    """
    Node IDs of the mesh, in the same order as meshio's points.
    Reads the $Nodes block of an ASCII .msh; falls back to 1..N otherwise.
    """
    try:
        with open(mesh_path, "r", encoding="utf-8", errors="strict") as f:
            for line in f:
                if line.strip() == "$Nodes":
                    n = int(f.readline())
                    tags = np.empty(n, dtype=np.int64)
                    for i in range(n):
                        tags[i] = int(f.readline().split()[0])
                    if n == n_nodes:
                        return tags
                    break
    except (UnicodeDecodeError, ValueError):
        pass  # binary or unexpected layout
    return np.arange(1, n_nodes + 1)


def read_node_data_block(path):
    """
    Parse the first $NodeData block of a gmsh file.
    Layout:  $NodeData / <#string tags> + lines / <#real tags> + lines /
             <#int tags> + lines (int tags: step, n_components, n_nodes) /
             'node_id v1 [v2 ...]' lines / $EndNodeData
    Returns (node_ids, values) with values shape (n,) or (n, n_components).
    """
    with open(path, "r") as f:
        lines = iter(f)
        for line in lines:
            if line.strip() == "$NodeData" or line.strip() == "$ElementData":
                break
        else:
            return None

        for _ in range(int(next(lines))):   # string tags (e.g. "Potential (V)")
            next(lines)
        for _ in range(int(next(lines))):   # real tags (e.g. time)
            next(lines)
        n_int = int(next(lines))            # integer tags
        ints = [int(next(lines)) for _ in range(n_int)]
        n_comp, n_entries = ints[1], ints[2]

        ids = np.empty(n_entries, dtype=np.int64)
        vals = np.empty((n_entries, n_comp))
        for i in range(n_entries):
            parts = next(lines).split()
            ids[i] = int(parts[0])
            vals[i] = [float(p) for p in parts[1:1 + n_comp]]
    return ids, (vals[:, 0] if n_comp == 1 else vals)


def load_node_data(path, node_tags):
    """
    Return one value per mesh node (in mesh point order) from a gmsh
    $NodeData file, matching values to nodes by node ID.
    """
    block = read_node_data_block(path)
    if block is None:
        raise ValueError(f"No $NodeData block found in {path}")
    ids, vals = block
    if vals.ndim > 1:
        print(f"Data has {vals.shape[1]} components; plotting the magnitude.")
        vals = np.linalg.norm(vals, axis=1)

    lookup = dict(zip(ids.tolist(), vals))
    missing = [t for t in node_tags.tolist() if t not in lookup]
    if missing:
        print(f"Warning: {len(missing)} mesh nodes have no data (set to NaN).")
    return np.array([lookup.get(t, np.nan) for t in node_tags.tolist()])


def sample_plane(grid, z, xlim, ylim, nx, ny):
    """Interpolate the mesh data onto a regular grid at height z."""
    xs = np.linspace(xlim[0], xlim[1], nx)
    ys = np.linspace(ylim[0], ylim[1], ny)
    X, Y = np.meshgrid(xs, ys)  # shape (ny, nx)
    pts = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z)])

    probe = pv.PolyData(pts)
    sampled = probe.sample(grid)  # interpolates inside tetrahedra

    vals = np.asarray(sampled.point_data["values"], dtype=float)
    valid = np.asarray(sampled.point_data["vtkValidPointMask"]).astype(bool)
    vals[~valid] = np.nan  # outside the mesh
    return xs, ys, vals.reshape(ny, nx)


def plot_slice(xs, ys, img, z, title=None, out=None, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        img,
        origin="lower",  # row 0 = smallest y
        extent=[xs[0], xs[-1], ys[0], ys[-1]],
        cmap=cmap,
        aspect="equal",
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, label="value")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title or f"Slice at z = {z}")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=200)
        print(f"Saved {out}")
    plt.show()


def main(argv=None):
    p = argparse.ArgumentParser(description="Plot a z-slice of tetrahedral mesh data.")
    p.add_argument("mesh", nargs="?", default=DEFAULT_MESH, help="gmsh .msh file")
    p.add_argument("data", nargs="?", default=DEFAULT_DATA, help="per-node data file")
    p.add_argument("--z", type=float, default=DEFAULT_Z, help="z coordinate of the plane")
    p.add_argument("--xlim", type=float, nargs=2, help="x range (default: mesh bounds)")
    p.add_argument("--ylim", type=float, nargs=2, help="y range (default: mesh bounds)")
    p.add_argument("--nx", type=int, default=300, help="pixels in x")
    p.add_argument("--ny", type=int, default=300, help="pixels in y")
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--out", help="save figure to this file (e.g. slice.png)")
    args = p.parse_args(argv)
    #github check
    grid = load_mesh(args.mesh)
    print(grid)

    tags = get_node_tags(args.mesh, grid.n_points)
    grid.point_data["values"] = load_node_data(args.data, tags)

    xmin, xmax, ymin, ymax, zmin, zmax = grid.bounds
    if not (zmin <= args.z <= zmax):
        sys.exit(f"z = {args.z} is outside the mesh (z range: {zmin} to {zmax})")
    xlim = args.xlim or (xmin, xmax)
    ylim = args.ylim or (ymin, ymax)

    xs, ys, img = sample_plane(grid, args.z, xlim, ylim, args.nx, args.ny)
    plot_slice(xs, ys, img, args.z, out=args.out, cmap=args.cmap)


if __name__ == "__main__":
    main()