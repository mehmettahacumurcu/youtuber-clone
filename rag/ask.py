"""Canonical Speaker CLI: verified RAG by default, explicit free generation with --ungrounded."""
from __future__ import annotations

import argparse

from pipeline.config import load_settings
from rag.ask_finetuned import answer as answer_finetuned


def answer(
    question: str,
    ungrounded: bool,
    config: str | None = None,
    *,
    ensure_absent_fn=None,
) -> str:
    settings = load_settings(config)
    return answer_finetuned(
        question,
        model=settings.rag.ollama_model,
        use_rag=not ungrounded,
        config=config,
        ensure_absent_fn=ensure_absent_fn,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask Speaker with verified clean-transcript RAG.")
    parser.add_argument("question")
    parser.add_argument(
        "--ungrounded",
        action="store_true",
        help="disable RAG before generation; this is explicit free generation, not a fallback",
    )
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    print(answer(args.question, ungrounded=args.ungrounded, config=args.config))


if __name__ == "__main__":
    main()
