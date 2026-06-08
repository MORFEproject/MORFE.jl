import meshio
import plotly.graph_objects as go
from pathlib import Path

# Get script directory as a Path object
script_dir = Path(__file__).resolve().parent

mesh = meshio.read(script_dir / "mesh.vtu")

pts = mesh.points
pd = mesh.point_data
node_id = pd["node_id"]
x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
dof_x, dof_y, dof_z = pd["dof_x"], pd["dof_y"], pd["dof_z"]

fig = go.Figure(go.Scatter3d(
    x=x, y=y, z=z,
    mode="markers",
    marker=dict(size=2, color=z, colorscale="Viridis", showscale=True,
                colorbar=dict(title="z")),
    text=[
        f"node_id={nid}<br>x={xi:.6f}<br>y={yi:.6f}<br>z={zi:.6f}<br>dof_x={dx}<br>dof_y={dy}<br>dof_z={dz}"
        for nid, xi, yi, zi, dx, dy, dz in zip(node_id, x, y, z, dof_x, dof_y, dof_z)
    ],
    hoverinfo="text",
))

fig.update_layout(
    title="Mesh nodes — hover to read node_id, coordinates and DOFs",
    scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="data"),
    margin=dict(l=0, r=0, t=40, b=0),
)

fig.write_html("mesh_viewer.html")
import webbrowser, pathlib
webbrowser.open(pathlib.Path("mesh_viewer.html").resolve().as_uri())
