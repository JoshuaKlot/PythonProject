"""
Tkinter viewer: plot a 2D slice (constant z) of node-based data defined on an unstructured
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
  5. matplotlib imshow, embedded in a tkinter window with a z slider, draws the result (points outside the mesh = NaN/blank)
"""
import argparse
import sys

import tkinter as tk
from tkinter import ttk

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import meshio
import numpy as np
import pyvista as pv

# ----------------------------------------------------------------------------
# DEFAULTS (used when no command-line arguments are given, e.g. run in PyCharm)
# ----------------------------------------------------------------------------
DEFAULT_MESH = "Mesh slice project/Charging Mesh Results/mesh1.msh"
DEFAULT_DATA = "Mesh slice project/Charging Mesh Results/MeshShape_potential.msh"  # <- change me
DEFAULT_Z = 0.0
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
            if line.strip() == "$NodeData":
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


class SliceSampler:
    """Interpolates the mesh data onto a regular (x, y) grid at a given z."""

    def __init__(self, grid, xlim, ylim, nx, ny):
        self.grid = grid
        self.xs = np.linspace(xlim[0], xlim[1], nx)
        self.ys = np.linspace(ylim[0], ylim[1], ny)
        X, Y = np.meshgrid(self.xs, self.ys)  # shape (ny, nx)
        self.X, self.Y = X.ravel(), Y.ravel()
        self.shape = (ny, nx)

        # Per-tetrahedron z range, so each slice only probes the cells the
        # plane actually crosses (much faster than probing the whole mesh).
        tets = grid.cells.reshape(-1, 5)[:, 1:]
        tz = grid.points[tets, 2]
        self.cell_zmin, self.cell_zmax = tz.min(axis=1), tz.max(axis=1)

    def sample(self, z):
        idx = np.flatnonzero((self.cell_zmin <= z) & (self.cell_zmax >= z))
        if idx.size == 0:
            return np.full(self.shape, np.nan)
        sub = self.grid.extract_cells(idx)
        pts = np.column_stack([self.X, self.Y, np.full(self.X.size, z)])
        sampled = pv.PolyData(pts).sample(sub)
        vals = np.asarray(sampled.point_data["values"], dtype=float)
        valid = np.asarray(sampled.point_data["vtkValidPointMask"]).astype(bool)
        vals[~valid] = np.nan  # outside the mesh
        return vals.reshape(self.shape)


class SliceApp(tk.Tk):
    """Tkinter window: matplotlib plot + z slider, entry box and step buttons."""

    SLIDER_STEPS = 1000  # resolution of the slider

    def __init__(self, sampler, z0, zmin, zmax, vmin, vmax, cmap="viridis", autoscale=True):
        super().__init__()
        self.title("Mesh slice viewer")
        self.sampler = sampler
        self.zmin, self.zmax = zmin, zmax
        self.vmin, self.vmax = vmin, vmax
        self._after_id = None
        self._updating = False  # guards slider <-> entry feedback loops

        # ---- matplotlib figure embedded in the window ----
        self.fig = Figure(figsize=(7, 6), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.im = self.ax.imshow(
            sampler.sample(z0), origin="lower", cmap=cmap, aspect="equal",
            interpolation="nearest",
            extent=[sampler.xs[0], sampler.xs[-1], sampler.ys[0], sampler.ys[-1]],
            vmin=vmin, vmax=vmax,
        )
        self.cbar = self.fig.colorbar(self.im, ax=self.ax, label="value")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_title(f"Slice at z = {z0:.4g}")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)

        # ---- controls (packed first so they are never squeezed off-screen) ----
        ctrl = ttk.Frame(self, padding=6)
        ctrl.pack(side=tk.BOTTOM, fill=tk.X)
        NavigationToolbar2Tk(self.canvas, self, pack_toolbar=True)  # zoom / save
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        ttk.Label(ctrl, text="z:").pack(side=tk.LEFT)
        self.z_var = tk.StringVar(value=f"{z0:.4g}")
        entry = ttk.Entry(ctrl, textvariable=self.z_var, width=10)
        entry.pack(side=tk.LEFT, padx=4)
        entry.bind("<Return>", self._on_entry)

        ttk.Button(ctrl, text="<", width=3, command=lambda: self._nudge(-1)).pack(side=tk.LEFT)
        self.scale = ttk.Scale(ctrl, from_=0, to=self.SLIDER_STEPS, orient=tk.HORIZONTAL,
                               command=self._on_slide)
        self.scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(ctrl, text=">", width=3, command=lambda: self._nudge(+1)).pack(side=tk.LEFT)

        self.auto_var = tk.BooleanVar(value=autoscale)
        ttk.Checkbutton(ctrl, text="Autoscale colors", variable=self.auto_var,
                        command=self._refresh).pack(side=tk.LEFT, padx=8)

        self.set_z(z0, refresh=False)
        self.bind("<Left>", lambda e: self._nudge(-1))
        self.bind("<Right>", lambda e: self._nudge(+1))

    # ---- z <-> slider mapping ----
    def _z_to_pos(self, z):
        return (z - self.zmin) / (self.zmax - self.zmin) * self.SLIDER_STEPS

    def _pos_to_z(self, pos):
        return self.zmin + float(pos) / self.SLIDER_STEPS * (self.zmax - self.zmin)

    def set_z(self, z, refresh=True):
        z = min(max(z, self.zmin), self.zmax)
        self._updating = True
        self.z_var.set(f"{z:.4g}")
        self.scale.set(self._z_to_pos(z))
        self._updating = False
        if refresh:
            self._schedule_refresh()

    # ---- event handlers ----
    def _on_slide(self, pos):
        if self._updating:
            return
        self.z_var.set(f"{self._pos_to_z(pos):.4g}")
        self._schedule_refresh()

    def _on_entry(self, _event=None):
        try:
            self.set_z(float(self.z_var.get()))
        except ValueError:
            self.bell()

    def _nudge(self, direction):
        step = (self.zmax - self.zmin) / 200.0
        self.set_z(self._pos_to_z(self.scale.get()) + direction * step)

    def _schedule_refresh(self):
        # Debounce: redraw ~100 ms after the last change while dragging.
        if self._after_id is not None:
            self.after_cancel(self._after_id)
        self._after_id = self.after(100, self._refresh)

    def _refresh(self):
        self._after_id = None
        try:
            z = float(self.z_var.get())
        except ValueError:
            return
        img = self.sampler.sample(z)
        self.im.set_data(img)
        if self.auto_var.get() and np.isfinite(img).any():
            self.im.set_clim(np.nanmin(img), np.nanmax(img))
        else:
            self.im.set_clim(self.vmin, self.vmax)
        self.ax.set_title(f"Slice at z = {z:.4g}")
        self.canvas.draw_idle()


def main(argv=None):
    p = argparse.ArgumentParser(description="Tkinter z-slice viewer for tetrahedral mesh data.")
    p.add_argument("mesh", nargs="?", default=DEFAULT_MESH, help="gmsh .msh file")
    p.add_argument("data", nargs="?", default=DEFAULT_DATA, help="per-node data file")
    p.add_argument("--z", type=float, default=DEFAULT_Z, help="starting z of the plane")
    p.add_argument("--xlim", type=float, nargs=2, help="x range (default: mesh bounds)")
    p.add_argument("--ylim", type=float, nargs=2, help="y range (default: mesh bounds)")
    p.add_argument("--nx", type=int, default=300, help="pixels in x")
    p.add_argument("--ny", type=int, default=300, help="pixels in y")
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--autoscale", action="store_true",
                   help="start with per-slice color scaling (default: fixed global range)")
    args = p.parse_args(argv)

    grid = load_mesh(args.mesh)
    print(grid)

    tags = get_node_tags(args.mesh, grid.n_points)
    grid.point_data["values"] = load_node_data(args.data, tags)

    xmin, xmax, ymin, ymax, zmin, zmax = grid.bounds
    z0 = min(max(args.z, zmin), zmax)
    vals = grid.point_data["values"]
    sampler = SliceSampler(grid, args.xlim or (xmin, xmax), args.ylim or (ymin, ymax),
                           args.nx, args.ny)
    app = SliceApp(sampler, z0, zmin, zmax, np.nanmin(vals), np.nanmax(vals),
                   cmap=args.cmap, autoscale=args.autoscale)
    app.mainloop()


if __name__ == "__main__":
    main()