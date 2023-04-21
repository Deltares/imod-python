import numpy as np
import pytest
import xugrid as xu

import imod


def create_quadgrid(ibound):
    return xu.Ugrid2d.from_structured(ibound)


def create_trigrid(ibound):
    import matplotlib

    dx, xmin, xmax, dy, ymin, ymax = imod.util.spatial_reference(ibound)
    x = np.arange(xmin, xmax + dx, dx)
    y = np.arange(ymin, ymax + abs(dy), abs(dy))
    node_y, node_x = [a.ravel() for a in np.meshgrid(x, y, indexing="ij")]
    triangulation = matplotlib.tri.Triangulation(node_x, node_y)
    grid = xu.Ugrid2d(node_x, node_y, -1, triangulation.triangles)
    return grid


@pytest.mark.usefixtures("imodflow_model")
@pytest.mark.parametrize("create_grid", [create_quadgrid, create_trigrid])
def test_convert_to_disv(imodflow_model, tmp_path, create_grid):
    imodflow_model.write(tmp_path / "imodflow")

    data, repeats = imod.prj.open_projectfile_data(tmp_path / "imodflow/imodflow.prj")
    tim_data = imod.prj.read_timfile(tmp_path / "imodflow/time_discretization.tim")
    times = sorted([d["time"] for d in tim_data])
    target = create_grid(data["bnd"]["ibound"])

    disv_model = imod.prj.convert_to_disv(
        projectfile_data=data,
        target=target,
        time_min=times[0],
        time_max=times[-1],
        repeat_stress=repeats,
    )

    simulation = imod.mf6.Modflow6Simulation(name="disv")
    simulation["gwf"] = disv_model
    simulation["solver"] = imod.mf6.Solution(
        modelnames=["gwf"],
        print_option="summary",
        csv_output=False,
        no_ptc=True,
        outer_dvclose=1.0e-4,
        outer_maximum=500,
        under_relaxation=None,
        inner_dvclose=1.0e-4,
        inner_rclose=0.001,
        inner_maximum=100,
        linear_acceleration="cg",
        scaling_method=None,
        reordering_method=None,
        relaxation_factor=0.97,
    )
    simulation.create_time_discretization(times)

    modeldir = tmp_path / "disv"
    simulation.write(modeldir)
    simulation.run()

    head = imod.mf6.open_hds(modeldir / "gwf/gwf.hds", modeldir / "gwf/disv.disv.grb")
    assert isinstance(head, xu.UgridDataArray)
