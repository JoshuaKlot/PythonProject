import meshio

mesh = meshio.read(
    "Mesh slice project\Charging Mesh Results\MeshShape_charge_density.msh",  # string, os.PathLike, or a buffer/open file
    # file_format="stl",  # optional if filename is a path; inferred from extension
    # see meshio-convert -h for all possible formats
)

if __name__ == "__main__":
    print(mesh)