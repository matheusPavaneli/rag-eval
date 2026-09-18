import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from rageval.config import get_settings
from rageval.eval.answers import AnswerReport
from rageval.eval.golden import GoldenSetError, load_golden_set
from rageval.eval.runner import EvalReport
from rageval.ingest.documents import DocumentReadError, load_documents
from rageval.site import SiteExportError, build_site, load_report, write_site

REPORTS = Path("evals/reports")
ANSWER_BASELINE = REPORTS / "20260918T175155Z-26b03ce9a1c2c1d4-answer-dense.json"
GATED = (
    REPORTS / "20260918T175148Z-26b03ce9a1c2c1d4-dense.json",
    REPORTS / "20260918T175145Z-26b03ce9a1c2c1d4-bm25.json",
    REPORTS / "20260918T175152Z-26b03ce9a1c2c1d4-hybrid-bm25.json",
)
RECORDED = (
    REPORTS / "20260918T134846Z-26b03ce9a1c2c1d4-fulltext.json",
    REPORTS / "20260918T134850Z-26b03ce9a1c2c1d4-hybrid-fulltext.json",
    REPORTS / "20260918T141727Z-26b03ce9a1c2c1d4-rerank-dense.json",
    REPORTS / "20260918T141752Z-26b03ce9a1c2c1d4-rerank-hybrid-bm25.json",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rageval.site",
        description="Export the gated eval reports and their documents as static data for web/",
    )
    parser.add_argument("--out", type=Path, required=True, help="directory to write JSON into")
    parser.add_argument("--answers", type=Path, default=ANSWER_BASELINE)
    parser.add_argument(
        "--gated", type=Path, nargs="*", default=list(GATED), help="reports CI compares against"
    )
    parser.add_argument(
        "--recorded", type=Path, nargs="*", default=list(RECORDED), help="reports not gated"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    settings = get_settings()
    arguments = build_parser().parse_args(argv)
    try:
        documents = load_documents(settings.documents_dir)
        questions = load_golden_set(settings.golden_set_path, documents)
        retrieval = [
            (path.name, load_report(path, EvalReport), gated)
            for paths, gated in ((arguments.gated, True), (arguments.recorded, False))
            for path in paths
        ]
        site = build_site(
            documents,
            questions,
            (arguments.answers.name, load_report(arguments.answers, AnswerReport)),
            retrieval,
        )
        write_site(site, arguments.out)
    except (SiteExportError, GoldenSetError, DocumentReadError, OSError) as error:
        print(f"site export failed: {error}", file=sys.stderr)
        return 1

    payload = site.payload
    print(
        f"wrote {len(site.pages)} questions, {len(payload.configurations)} configurations, "
        f"{payload.resolved_count} resolved of {payload.citation_count} citations "
        f"to {arguments.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
