# flake8: noqa: B008

import logging
from pathlib import Path

import trimesh
import typer

from yumo.__about__ import __application__
from yumo.app import Config, PolyscopeApp
from yumo.constants import DATA_PREPROCESS_METHODS
from yumo.utils import load_mesh, parse_plt_file, write_plt_file

app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]})


def configure_logging(log_level: str):
    # Map text log level to numeric
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise typer.BadParameter(f"Invalid log level: {log_level}")

    # By default, root logger follows the chosen level
    root_level = numeric_level

    # Special case: if DEBUG is chosen, don't expose 3rd-party debug
    if numeric_level == logging.DEBUG:
        root_level = logging.INFO

    # Configure root logger
    logging.basicConfig(level=root_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Configure application logger separately
    app_logger = logging.getLogger(__application__)
    if numeric_level == logging.DEBUG:
        app_logger.setLevel(logging.DEBUG)
    else:
        app_logger.setLevel(numeric_level)


@app.command()
def prune(
    data_path: Path = typer.Option(
        ...,
        "--data",
        "-d",
        help="Path to data file (e.g. Tecplot .plt file)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    data_out_path: Path = typer.Option(
        ...,
        "--data-out",
        "-o",
        help="Path to output data file (e.g. Tecplot .plt file)",
        file_okay=True,
        dir_okay=False,
        writable=True,
    ),
    mesh_path: Path = typer.Option(
        ...,
        "--mesh",
        "-m",
        help="Path to mesh file (e.g. .stl file)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
):
    """Prune data points that are inside the mesh."""
    typer.echo(f"Loading points from {data_path} ...")
    points = parse_plt_file(data_path, skip_zeros=False)

    typer.echo(f"Loading mesh from {mesh_path} ...")
    mesh: trimesh.Trimesh = load_mesh(mesh_path, return_trimesh=True)  # type: ignore[assignment]
    if not mesh.is_watertight:
        typer.echo("Mesh is not watertight, skipping.")
        return

    typer.echo("Checking which points are inside mesh ...")
    inside_mask = mesh.contains(points[:, :3])

    typer.echo(f"Found {inside_mask.sum()} points inside the mesh, pruning ...")
    kept_points = points[~inside_mask]

    typer.echo(f"Writing {len(kept_points)} points to {data_out_path} ...")
    write_plt_file(data_out_path, kept_points)

    typer.echo("Done.")


@app.command()
def viz(
    data_path: Path = typer.Option(
        ...,
        "--data",
        "-d",
        help="Path to data file (e.g. Tecplot .plt file)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    mesh_path: Path | None = typer.Option(
        None,
        "--mesh",
        "-m",
        help="Path to mesh file (e.g. .stl file)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    sample_rate: float = typer.Option(
        1.0,
        "--sample-rate",
        "-s",
        min=0.0,
        max=1.0,
        help="Sampling rate for large datasets (0.0-1.0)",
    ),
    preprocess_method: str = typer.Option(
        "identity",
        "--preprocess-method",
        "--prep",
        help=f"Method to preprocess data. One of {DATA_PREPROCESS_METHODS}",
    ),
    camera_view: Path | None = typer.Option(
        None,
        "--camera-view",
        "--cam",
        help="Path to camera view json file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    skip_zeros: bool = typer.Option(
        False,
        help="Skip loading points with values = 0.0",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    ),
) -> None:
    """Visualize the scalar field."""
    # Configure logging based on the provided log_level
    configure_logging(log_level)

    PolyscopeApp(
        config=Config(
            data_path=data_path,
            mesh_path=mesh_path,
            sample_rate=sample_rate,
            skip_zeros=skip_zeros,
            data_preprocess_method=preprocess_method,
            camera_view_path=camera_view,
        )
    ).run()


def main():
    app()


if __name__ == "__main__":
    main()
