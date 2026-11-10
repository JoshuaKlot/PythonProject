"""
Tkinter viewer: plot a 2D slice of node-based data defined on an unstructured
tetrahedral gmsh mesh (.msh, version 2.2).

The slice plane can be perpendicular to ANY of the three axes:
    axis = x  -> plane at constant x, image shows (y, z)
    axis = y  -> plane at constant y, image shows (x, z)
    axis = z  -> plane at constant z, image shows (x, y)

Usage (command line):
    python mesh_slice.py mesh1.msh data.txt --axis z --pos 0.5
    python mesh_slice.py mesh1.msh data.txt --axis x --pos 0.2 --ylim -1 1 --zlim 0 2
    python mesh_slice.py mesh1.msh data.txt --z 0.5          # old-style shortcut for --axis z --pos 0.5

Or from PyCharm: edit the DEFAULTS block below and press run.

Pipeline:
  1. meshio reads the .msh  -> nodes + tetrahedra
  2. the data file is read   -> one value per node
  3. PyVista builds an UnstructuredGrid, attaches the data to the nodes
  4. a regular 2D grid lying in the chosen plane is sampled from the mesh
     (PyVista interpolates inside each tetrahedron)
  5. matplotlib imshow, embedded in a tkinter window with file pickers, an
     axis selector and a position slider, draws the result
     (points outside the mesh = NaN/blank)
"""
import argparse
import os

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

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
DEFAULT_DATA = "Mesh slice project/Charging Mesh Results/MeshShape_potential.msh"
DEFAULT_AXIS = "z"    # which axis the slice plane is perpendicular to: "x", "y" or "z"
DEFAULT_POS = 0.0     # starting position of the plane along that axis
# ----------------------------------------------------------------------------

# Map axis letter -> column index in the (N, 3) points array.
AXES = {"x": 0, "y": 1, "z": 2}


def in_plane_axes(axis_idx):
    """
    Return the two axis indices that remain in the image when slicing
    perpendicular to `axis_idx`, as (horizontal, vertical).
        slice x -> (y, z)     slice y -> (x, z)     slice z -> (x, y)
    """
    h, v = [i for i in range(3) if i != axis_idx]
    return h, v


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
    if len(ids) != len(node_tags):
        raise ValueError(
            f"The data file has {len(ids)} node values but the mesh has "
            f"{len(node_tags)} nodes. This data does not belong to this mesh."
        )
    if vals.ndim > 1:
        print(f"Data has {vals.shape[1]} components; plotting the magnitude.")
        vals = np.linalg.norm(vals, axis=1)

    lookup = dict(zip(ids.tolist(), vals))
    missing = [t for t in node_tags.tolist() if t not in lookup]
    if missing:
        print(f"Warning: {len(missing)} mesh nodes have no data (set to NaN).")
    return np.array([lookup.get(t, np.nan) for t in node_tags.tolist()])


class SliceSampler:
    """
    Interpolates the mesh data onto a regular 2D grid lying in a plane that is
    perpendicular to one coordinate axis.

    axis_idx : index (0=x, 1=y, 2=z) of the axis the plane is perpendicular to
    h_lim    : (min, max) of the horizontal in-plane axis
    v_lim    : (min, max) of the vertical in-plane axis
    nx, ny   : number of pixels horizontally / vertically
    """

    def __init__(self, grid, axis_idx, h_lim, v_lim, nx, ny):
        self.grid = grid
        self.axis_idx = axis_idx
        self.h_idx, self.v_idx = in_plane_axes(axis_idx)

        # Regular pixel grid in the plane's own 2D coordinates.
        hs = np.linspace(h_lim[0], h_lim[1], nx)
        vs = np.linspace(v_lim[0], v_lim[1], ny)
        H, V = np.meshgrid(hs, vs)  # shape (ny, nx)
        self.H, self.V = H.ravel(), V.ravel()
        self.shape = (ny, nx)

        # Per-tetrahedron range along the slice axis, so each slice only probes
        # the cells the plane actually crosses (much faster than probing the
        # whole mesh).
        tets = grid.cells.reshape(-1, 5)[:, 1:]
        ta = grid.points[tets, axis_idx]
        self.cell_min, self.cell_max = ta.min(axis=1), ta.max(axis=1)

    def sample(self, pos):
        """Return a (ny, nx) image of the data on the plane at coordinate `pos`."""
        idx = np.flatnonzero((self.cell_min <= pos) & (self.cell_max >= pos))
        if idx.size == 0:
            return np.full(self.shape, np.nan)
        sub = self.grid.extract_cells(idx)

        # Build 3D sample points: put the 2D pixel coordinates in the right
        # columns and fill the slice-axis column with the constant `pos`.
        pts = np.empty((self.H.size, 3))
        pts[:, self.h_idx] = self.H
        pts[:, self.v_idx] = self.V
        pts[:, self.axis_idx] = pos

        sampled = pv.PolyData(pts).sample(sub)
        vals = np.asarray(sampled.point_data["values"], dtype=float)
        valid = np.asarray(sampled.point_data["vtkValidPointMask"]).astype(bool)
        vals[~valid] = np.nan  # outside the mesh
        return vals.reshape(self.shape)


class SliceApp(tk.Tk):
    """
    Tkinter window: file pickers (mesh + data), matplotlib plot, an axis
    selector (X / Y / Z), and a position slider with entry box and step buttons.
    """

    SLIDER_STEPS = 1000  # resolution of the slider

    def __init__(self, mesh_path, data_path, axis="z", pos0=0.0, limits=None,
                 nx=300, ny=300, cmap="viridis", autoscale=False):
        super().__init__()
        self.title("Mesh slice viewer")
        self.nx, self.ny = nx, ny
        # Command-line in-plane limits, one (min, max) or None per axis [x, y, z].
        # They apply to the first mesh only.
        self.start_limits = list(limits) if limits else [None, None, None]
        self.pending_pos = pos0  # starting plane position, consumed on first mesh load

        self.grid = None        # PyVista grid of the current mesh
        self.tags = None        # node IDs of the current mesh
        self.limits = None      # current (min, max) for each of x, y, z
        self.samplers = {}      # cache: axis letter -> SliceSampler
        self.sampler = None     # sampler for the currently selected axis
        self.has_data = False
        self.pmin, self.pmax = 0.0, 1.0   # range of the plane position along the active axis
        self.vmin = self.vmax = None      # global colour range of the data
        self._after_id = None
        self._updating = False  # guards slider <-> entry feedback loops

        # ---- file pickers ----
        files = ttk.Frame(self, padding=(6, 6, 6, 0))
        files.pack(side=tk.TOP, fill=tk.X)
        files.columnconfigure(1, weight=1)
        self.mesh_var = tk.StringVar(value=mesh_path)
        self.data_var = tk.StringVar(value=data_path)
        rows = [
            ("Mesh:", self.mesh_var, self.browse_mesh, self.load_mesh_file),
            ("Data:", self.data_var, self.browse_data, self.load_data_file),
        ]
        for r, (label, var, browse, load) in enumerate(rows):
            ttk.Label(files, text=label).grid(row=r, column=0, sticky="w")
            entry = ttk.Entry(files, textvariable=var)
            entry.grid(row=r, column=1, sticky="ew", padx=4, pady=2)
            entry.bind("<Return>", lambda _e, f=load: f())
            ttk.Button(files, text="Browse...", command=browse).grid(row=r, column=2)
            ttk.Button(files, text="Load", command=load).grid(row=r, column=3, padx=(4, 0))
        self.status_var = tk.StringVar(value="")
        ttk.Label(files, textvariable=self.status_var, foreground="gray35").grid(
            row=2, column=0, columnspan=4, sticky="w")

        # ---- matplotlib figure embedded in the window ----
        self.fig = Figure(figsize=(7, 6), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.im = self.ax.imshow(
            np.full((ny, nx), np.nan), origin="lower", cmap=cmap, aspect="equal",
            interpolation="nearest", extent=[0, 1, 0, 1],
        )
        self.cbar = self.fig.colorbar(self.im, ax=self.ax, label="value")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_title("No data loaded")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)

        # ---- controls (packed before the canvas so they are never squeezed off-screen) ----
        # Row 1 (bottom-most): axis selector
        axis_row = ttk.Frame(self, padding=(6, 0, 6, 6))
        axis_row.pack(side=tk.BOTTOM, fill=tk.X)
        # Row 2: position entry, slider, step buttons, autoscale
        ctrl = ttk.Frame(self, padding=6)
        ctrl.pack(side=tk.BOTTOM, fill=tk.X)
        NavigationToolbar2Tk(self.canvas, self, pack_toolbar=True)  # zoom / save
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Axis selector: radio buttons share one StringVar holding "x", "y" or "z".
        ttk.Label(axis_row, text="Slice axis (plane is perpendicular to):").pack(side=tk.LEFT)
        self.axis_var = tk.StringVar(value=axis)
        for letter in ("x", "y", "z"):
            ttk.Radiobutton(axis_row, text=letter.upper(), value=letter,
                            variable=self.axis_var,
                            command=self._on_axis).pack(side=tk.LEFT, padx=6)

        # Position entry + slider. The label text follows the selected axis ("x:", "y:", "z:").
        self.pos_label_var = tk.StringVar(value=f"{axis}:")
        ttk.Label(ctrl, textvariable=self.pos_label_var, width=3).pack(side=tk.LEFT)
        self.pos_var = tk.StringVar(value=f"{pos0:.4g}")
        pos_entry = ttk.Entry(ctrl, textvariable=self.pos_var, width=10)
        pos_entry.pack(side=tk.LEFT, padx=4)
        pos_entry.bind("<Return>", self._on_entry)

        ttk.Button(ctrl, text="<", width=3, command=lambda: self._nudge(-1)).pack(side=tk.LEFT)
        self.scale = ttk.Scale(ctrl, from_=0, to=self.SLIDER_STEPS, orient=tk.HORIZONTAL,
                               command=self._on_slide)
        self.scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(ctrl, text=">", width=3, command=lambda: self._nudge(+1)).pack(side=tk.LEFT)

        self.auto_var = tk.BooleanVar(value=autoscale)
        ttk.Checkbutton(ctrl, text="Autoscale colors", variable=self.auto_var,
                        command=self._refresh).pack(side=tk.LEFT, padx=8)

        self.bind("<Left>", lambda e: self._nudge(-1))
        self.bind("<Right>", lambda e: self._nudge(+1))

        # Load the initial files after the window has appeared.
        self.after(50, self.load_mesh_file)

    # ------------------------------------------------------------------
    # file handling
    # ------------------------------------------------------------------
    @staticmethod
    def _start_dir(var):
        d = os.path.dirname(os.path.abspath(var.get())) if var.get().strip() else ""
        return d if os.path.isdir(d) else os.getcwd()

    def _busy(self, message):
        """Show a wait cursor + status message (or clear it when message is None)."""
        self.config(cursor="watch" if message else "")
        if message:
            self.status_var.set(message)
        self.update_idletasks()

    def browse_mesh(self):
        path = filedialog.askopenfilename(
            title="Select mesh file", initialdir=self._start_dir(self.mesh_var),
            filetypes=[("Gmsh mesh", "*.msh"), ("All files", "*.*")])
        if path:
            self.mesh_var.set(path)
            self.load_mesh_file()

    def browse_data(self):
        path = filedialog.askopenfilename(
            title="Select data file", initialdir=self._start_dir(self.data_var),
            filetypes=[("Gmsh data", "*.msh"), ("Text files", "*.txt *.dat"), ("All files", "*.*")])
        if path:
            self.data_var.set(path)
            self.load_data_file()

    def load_mesh_file(self):
        """Load the mesh in the Mesh box, then try the file in the Data box."""
        path = self.mesh_var.get().strip()
        if not path:
            return
        self._busy("Loading mesh...")
        try:
            grid = load_mesh(path)
            tags = get_node_tags(path, grid.n_points)
        except Exception as exc:
            self._busy(None)
            self.status_var.set("Mesh load failed.")
            messagebox.showerror("Could not load mesh", f"{path}\n\n{exc}")
            return

        # In-plane display limits for each axis: the command-line value if given
        # (first mesh only), otherwise the full mesh extent along that axis.
        b = grid.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
        self.limits = [self.start_limits[i] or (b[2 * i], b[2 * i + 1]) for i in range(3)]
        self.start_limits = [None, None, None]  # command-line limits apply once

        self.grid, self.tags = grid, tags
        self.has_data = False   # old data no longer matches this mesh
        self.samplers = {}      # cached samplers belong to the old mesh

        # Choose the starting plane position: command-line value on the first
        # load, otherwise whatever is in the entry box, otherwise the middle.
        axis = self.axis_var.get()
        a = AXES[axis]
        try:
            pos = self.pending_pos if self.pending_pos is not None else float(self.pos_var.get())
        except ValueError:
            pos = 0.5 * (b[2 * a] + b[2 * a + 1])
        self.pending_pos = None
        self.apply_axis(axis, pos=pos, refresh=False)

        self.status_var.set(f"Mesh: {grid.n_points} nodes, {grid.n_cells} tetrahedra.")
        if self.data_var.get().strip():
            self.load_data_file()
        else:
            self._busy(None)
            self._refresh()

    def load_data_file(self):
        """Load the per-node data in the Data box for the current mesh."""
        if self.grid is None:
            self.status_var.set("Load a mesh first.")
            return
        path = self.data_var.get().strip()
        if not path:
            return
        self._busy("Loading data...")
        try:
            values = load_node_data(path, self.tags)
        except Exception as exc:
            self._busy(None)
            self.has_data = False
            self._refresh()
            self.status_var.set("Data load failed.")
            messagebox.showerror("Could not load data", f"{path}\n\n{exc}")
            return

        self.grid.point_data["values"] = values
        self.vmin, self.vmax = np.nanmin(values), np.nanmax(values)
        self.has_data = True
        self._busy(None)
        self.status_var.set(
            f"Mesh: {self.grid.n_points} nodes, {self.grid.n_cells} tetrahedra. "
            f"Data range: {self.vmin:.4g} to {self.vmax:.4g}.")
        self._refresh()

    # ------------------------------------------------------------------
    # axis selection
    # ------------------------------------------------------------------
    def apply_axis(self, axis, pos=None, refresh=True):
        """
        Switch the slice axis ("x", "y" or "z"): rebuild/reuse the sampler,
        update the image extent and axis labels, reset the slider range to the
        mesh extent along the new axis, and place the plane at `pos`
        (default: the middle of the mesh along that axis).
        """
        if self.grid is None:
            return
        a = AXES[axis]
        h, v = in_plane_axes(a)
        names = "xyz"

        # Range the slider can cover = full mesh extent along the slice axis.
        self.pmin, self.pmax = self.grid.bounds[2 * a], self.grid.bounds[2 * a + 1]

        # Samplers are cached per axis so switching back and forth is instant.
        if axis not in self.samplers:
            self.samplers[axis] = SliceSampler(
                self.grid, a, self.limits[h], self.limits[v], self.nx, self.ny)
        self.sampler = self.samplers[axis]

        # Image extent and axis labels follow the two in-plane axes.
        hl, vl = self.limits[h], self.limits[v]
        self.im.set_extent([hl[0], hl[1], vl[0], vl[1]])
        self.ax.set_xlim(hl)
        self.ax.set_ylim(vl)
        self.ax.set_xlabel(names[h])
        self.ax.set_ylabel(names[v])
        self.pos_label_var.set(f"{axis}:")

        if pos is None:
            pos = 0.5 * (self.pmin + self.pmax)
        self.set_pos(pos, refresh=refresh)

    def _on_axis(self):
        """Radio button clicked: switch to the newly selected axis."""
        self.apply_axis(self.axis_var.get())

    # ------------------------------------------------------------------
    # position <-> slider mapping
    # ------------------------------------------------------------------
    def _pos_to_slider(self, p):
        span = self.pmax - self.pmin
        return 0.0 if span == 0 else (p - self.pmin) / span * self.SLIDER_STEPS

    def _slider_to_pos(self, s):
        return self.pmin + float(s) / self.SLIDER_STEPS * (self.pmax - self.pmin)

    def set_pos(self, p, refresh=True):
        """Clamp `p` to the mesh, update entry + slider, and (optionally) redraw."""
        p = min(max(p, self.pmin), self.pmax)
        self._updating = True   # stop the slider callback from echoing back
        self.pos_var.set(f"{p:.4g}")
        self.scale.set(self._pos_to_slider(p))
        self._updating = False
        if refresh:
            self._schedule_refresh()

    # ------------------------------------------------------------------
    # event handlers
    # ------------------------------------------------------------------
    def _on_slide(self, s):
        if self._updating:
            return
        self.pos_var.set(f"{self._slider_to_pos(s):.4g}")
        self._schedule_refresh()

    def _on_entry(self, _event=None):
        try:
            self.set_pos(float(self.pos_var.get()))
        except ValueError:
            self.bell()

    def _nudge(self, direction):
        """Move the plane by 1/200 of the mesh extent along the active axis."""
        step = (self.pmax - self.pmin) / 200.0
        self.set_pos(self._slider_to_pos(self.scale.get()) + direction * step)

    def _schedule_refresh(self):
        # Debounce: redraw ~100 ms after the last change while dragging.
        if self._after_id is not None:
            self.after_cancel(self._after_id)
        self._after_id = self.after(50, self._refresh)

    def _refresh(self):
        """Resample the mesh at the current plane position and redraw."""
        self._after_id = None
        if self.sampler is None or not self.has_data:
            self.im.set_data(np.full((self.ny, self.nx), np.nan))
            self.ax.set_title("No data loaded")
            self.canvas.draw_idle()
            return
        try:
            p = float(self.pos_var.get())
        except ValueError:
            return
        img = self.sampler.sample(p)
        self.im.set_data(img)
        # Colour scale: per-slice if autoscale is on, else fixed global range.
        if self.auto_var.get() and np.isfinite(img).any():
            self.im.set_clim(np.nanmin(img), np.nanmax(img))
        else:
            self.im.set_clim(self.vmin, self.vmax)
        self.ax.set_title(f"Slice at {self.axis_var.get()} = {p:.4g}")
        self.canvas.draw_idle()


def main(argv=None):
    p = argparse.ArgumentParser(description="Tkinter slice viewer for tetrahedral mesh data.")
    p.add_argument("mesh", nargs="?", default=DEFAULT_MESH, help="gmsh .msh file")
    p.add_argument("data", nargs="?", default=DEFAULT_DATA, help="per-node data file")
    p.add_argument("--axis", choices=["x", "y", "z"], default=DEFAULT_AXIS,
                   help="axis the slice plane is perpendicular to")
    p.add_argument("--pos", type=float, default=DEFAULT_POS,
                   help="starting position of the plane along --axis")
    p.add_argument("--z", type=float, default=None,
                   help="shortcut for --axis z --pos Z (kept for backwards compatibility)")
    p.add_argument("--xlim", type=float, nargs=2, help="x range (default: mesh bounds)")
    p.add_argument("--ylim", type=float, nargs=2, help="y range (default: mesh bounds)")
    p.add_argument("--zlim", type=float, nargs=2, help="z range (default: mesh bounds)")
    p.add_argument("--nx", type=int, default=300, help="pixels along the horizontal image axis")
    p.add_argument("--ny", type=int, default=300, help="pixels along the vertical image axis")
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--autoscale", action="store_true",
                   help="start with per-slice color scaling (default: fixed global range)")
    args = p.parse_args(argv)

    # Old-style --z overrides the axis/position pair.
    axis, pos = args.axis, args.pos
    if args.z is not None:
        axis, pos = "z", args.z

    limits = [args.xlim, args.ylim, args.zlim]
    app = SliceApp(args.mesh, args.data, axis=axis, pos0=pos, limits=limits,
                   nx=args.nx, ny=args.ny, cmap=args.cmap, autoscale=args.autoscale)
    app.mainloop()


if __name__ == "__main__":
    main()