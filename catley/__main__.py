"""Main entry point for the game."""

import argparse

from . import config
from .app import App, AppConfig
from .util import rng


def _parse_metric_names(raw: str | None) -> tuple[str, ...]:
    """Parse a comma-separated metric/variable list into canonical names."""
    if raw is None:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Catley.")
    parser.add_argument(
        "--metric-log",
        dest="metric_log_names",
        type=str,
        default=None,
        help=(
            "Comma-separated live variable/metric names to append periodically, "
            "e.g. 'time.render_ms,time.total_ms'."
        ),
    )
    parser.add_argument(
        "--metric-log-file",
        dest="metric_log_file",
        type=str,
        default="metric_samples.log",
        help=(
            "Output file for metric log rows (default: metric_samples.log). "
            "File is reset at startup when --metric-log is enabled."
        ),
    )
    parser.add_argument(
        "--metric-log-interval",
        dest="metric_log_interval_seconds",
        type=float,
        default=5.0,
        help="Seconds between metric log writes (default: 5.0).",
    )
    args = parser.parse_args()

    # Initialize the RNG stream system for deterministic randomness
    rng.init(config.RANDOM_SEED)

    metric_log_names = _parse_metric_names(args.metric_log_names)
    app_config = AppConfig(
        title=config.WINDOW_TITLE,
        width=config.SCREEN_WIDTH,
        height=config.SCREEN_HEIGHT,
        vsync=config.VSYNC,
        metric_log_names=metric_log_names,
        metric_log_file=args.metric_log_file if metric_log_names else None,
        metric_log_interval_seconds=args.metric_log_interval_seconds,
    )

    match config.BACKEND.app:
        case "glfw":
            from catley.backends.glfw.app import GlfwApp

            _APP_CLASS = GlfwApp

    app: App = _APP_CLASS(app_config)
    if metric_log_names:
        metric_log_file = args.metric_log_file
        print(
            "Metric logging enabled: "
            f"{','.join(metric_log_names)} -> {metric_log_file} "
            f"every {args.metric_log_interval_seconds:.2f}s"
        )
        print(f"Metric log file (reset on startup): {metric_log_file}")
    app.run()


if __name__ == "__main__":
    main()
