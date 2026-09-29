"""Shared command-line paths and error presentation."""

from contextlib import contextmanager

from fraise.paths import data_root


def add_data_dir(parser):
    parser.add_argument(
        "--data-dir",
        type=data_root,
        default=data_root(),
        help="data root (default: FRAISE_DATA_DIR or ./data)",
    )


@contextmanager
def cli_errors(parser):
    try:
        yield
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
