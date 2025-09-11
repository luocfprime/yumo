# flake8: noqa: B008

import logging
from pathlib import Path

import typer

from yumo.__about__ import __application__
from yumo.app import Config, PolyscopeApp

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
def _(
    data_path: Path = typer.Option(
        ...,
        "--data",
        help="Path to data file (e.g. Tecplot .plt file)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    mesh_path: Path | None = typer.Option(
        None,
        "--mesh",
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
    # Configure logging based on the provided log_level
    configure_logging(log_level)

    PolyscopeApp(
        config=Config(
            data_path=data_path,
            mesh_path=mesh_path,
            sample_rate=sample_rate,
            skip_zeros=skip_zeros,
        )
    ).run()


def main():
    app()


if __name__ == "__main__":
    main()
