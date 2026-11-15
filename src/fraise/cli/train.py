"""Train or resume a Fraise run from explicit configuration files."""

import argparse
from pathlib import Path

from fraise.config import load_model_config, load_training_settings
from fraise.config.compatibility import check_compatibility
from fraise.data.loading import load_training_data
from fraise.data.sources import SOURCE_NAMES
from fraise.tokenizer.loading import load_tokenizer
from fraise.training.run.artifacts import read_run_model_name
from fraise.training.training import pretrain_model
from fraise.training.events import AblationKitMetrics, EventWriter, emit, fan_out
from fraise.training.control import stop_signals

from .common import add_data_dir, cli_errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-hyperparameters", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--tokenizer-file", type=Path, required=True)
    parser.add_argument("--model-name")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument(
        "--stop-file", type=Path, help="save and stop when this file appears"
    )
    parser.add_argument(
        "--events-file", type=Path, help="append structured progress JSON lines"
    )
    parser.add_argument(
        "--ablationkit-metrics",
        action="store_true",
        help="log progress and validation metrics to the AblationKit job's run "
        "(for jobs that do not collect --events-file as their metrics file)",
    )
    parser.add_argument("--parent-pid", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--save-path", type=Path, help="run directory parent (default: DATA_DIR/models)"
    )
    add_data_dir(parser)
    args = parser.parse_args(argv)
    if args.parent_pid is not None:
        from fraise.training.validation.worker_lifetime import exit_with_parent

        exit_with_parent(args.parent_pid)
    events = fan_out(
        EventWriter(args.events_file) if args.events_file else None,
        AblationKitMetrics() if args.ablationkit_metrics else None,
    )
    with (
        cli_errors(parser),
        stop_signals(args.stop_file, events) as (on_event, stop_requested),
    ):
        emit(on_event, "stage", phase="loading")
        name = args.model_name or (
            read_run_model_name(args.resume) if args.resume else None
        )
        if not name:
            parser.error("--model-name is required for a new run")
        settings = load_training_settings(args.training_hyperparameters)
        config = load_model_config(args.model_config)
        settings.validate_sources(SOURCE_NAMES)
        tokenizer = load_tokenizer(args.tokenizer_file)
        check_compatibility(tokenizer, config, settings)
        data = load_training_data(
            settings.dataset_token_counts,
            tokenizer,
            root=args.data_dir,
            validation=settings.max_validation_tokens != 0,
        )
        if stop_requested():
            emit(on_event, "stopped", phase="loading")
            return
        pretrain_model(
            tokenizer,
            data,
            settings,
            config,
            args.save_path.expanduser() if args.save_path else args.data_dir / "models",
            name,
            resume_from=args.resume,
            config_paths={
                "training_hyperparameters": args.training_hyperparameters.expanduser(),
                "model_config": args.model_config.expanduser(),
            },
            device=args.device,
            stop_requested=stop_requested,
            on_event=on_event,
        )


if __name__ == "__main__":
    main()
