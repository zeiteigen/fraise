"""Serve the local Fraise model playground."""

import argparse
from pathlib import Path

from .common import add_data_dir, cli_errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_dir(parser)
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        import uvicorn
        from filelock import FileLock, Timeout

        from fraise.web.app import create_app
    except ImportError:
        parser.error("web dependencies are missing; run uv sync --extra web")
    args.data_dir.joinpath("web").mkdir(parents=True, exist_ok=True)
    with cli_errors(parser):
        try:
            with FileLock(str(args.data_dir / "web" / "server.lock"), timeout=0):
                app = create_app(
                    data_dir=args.data_dir,
                    models_dir=args.models_dir,
                    device=args.device,
                )
                uvicorn.run(app, host="127.0.0.1", port=args.port)
        except Timeout:
            parser.error("a Fraise web server is already using this data directory")


if __name__ == "__main__":
    main()
