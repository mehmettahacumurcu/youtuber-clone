"""Packaged Studio worker entry point."""
from __future__ import annotations

import argparse
import sys

from runtime.paths import RuntimePaths
from runtime.settings import RuntimeSettings


def run_package_healthcheck() -> None:
    """Import the packaged Studio graph without constructing or binding a server."""
    import ui.server  # noqa: F401
    import rag.evidence_pipeline  # noqa: F401
    import rag.verifier  # noqa: F401


def build_rag_command(
    settings: RuntimeSettings, *, executable: str | None = None, frozen: bool | None = None,
) -> list[str]:
    """Dispatch RAG through the frozen worker executable when Python modules are unavailable."""
    paths = RuntimePaths.from_settings(settings)
    executable = executable or sys.executable
    frozen = getattr(sys, "frozen", False) if frozen is None else frozen
    common = ["--port", str(settings.rag_port), "--config", str(paths.config_path), "--host", "127.0.0.1"]
    if frozen:
        return [executable, "--rag-worker", *common]
    return [executable, "-m", "rag.retrieve_server", *common]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--healthcheck", action="store_true")
    parser.add_argument("--rag-worker", action="store_true")
    parser.add_argument("--port", type=int)
    parser.add_argument("--config")
    parser.add_argument("--host")
    args = parser.parse_args()
    if args.healthcheck:
        try:
            run_package_healthcheck()
        except Exception as error:
            print(
                f"studio package healthcheck failed: {type(error).__name__}: {error}",
                file=sys.stderr,
            )
            return 1
        return 0
    settings = RuntimeSettings.from_environment()
    if args.rag_worker:
        if args.host != "127.0.0.1" or args.port != settings.rag_port or not args.config:
            parser.error("--rag-worker requires launcher-assigned loopback port and config")
        from rag.retrieve_server import main as rag_main

        return rag_main(["--port", str(args.port), "--config", args.config, "--host", args.host])
    import uvicorn
    from ui.server import create_app
    uvicorn.run(
        create_app(settings), host="127.0.0.1", port=settings.studio_port,
        access_log=False, log_config=None, proxy_headers=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
