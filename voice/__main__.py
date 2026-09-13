"""CLI entrypoint for the packaged voice worker."""
from __future__ import annotations

import argparse
import sys

import uvicorn

from runtime.settings import RuntimeSettings
from voice.server import create_app, run_package_healthcheck


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--healthcheck", action="store_true")
    args = parser.parse_args()
    settings = RuntimeSettings.from_environment()
    if args.healthcheck:
        try:
            run_package_healthcheck(settings)
        except Exception as error:
            print(
                f"voice package healthcheck failed: {type(error).__name__}: {error}",
                file=sys.stderr,
            )
            return 1
        return 0
    uvicorn.run(
        create_app(settings),
        host="127.0.0.1",
        port=settings.voice_port,
        log_config=None,
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
