# yumo

Scalar field visualization using Polyscope.

- **Git repository**: <https://github.com/luocfprime/yumo/>


- **Documentation** <https://luocfprime.github.io/yumo/>

## Install

Prerequisites: You must have at least one Python package manager installed (e.g. [uv](https://docs.astral.sh/uv/getting-started/installation/)).

Install it from PyPI:

```bash
uv tool install yumo
```

Or, if you want to run it once without installing it, you can use the `uv run` command:

```bash
uv run --with yumo yumo xxx  # xxx being the subcommand you want to run
```

## Usage

```text
$ yumo -h

 Usage: yumo [OPTIONS] COMMAND [ARGS]...

╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --install-completion            Install completion for the current shell.                                                                       │
│ --show-completion               Show completion for the current shell, to copy it or customize the installation.                                │
│ --help                -h        Show this message and exit.                                                                                     │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ prune   Prune data points that are inside the mesh.                                                                                             │
│ viz     Visualize the scalar field.                                                                                                             │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

MIT.
