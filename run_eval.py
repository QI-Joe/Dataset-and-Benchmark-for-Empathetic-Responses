"""
LLM-as-a-Judge G-Eval evaluator — main entry point.

Single-file mode  (input is a .json file):
    python run_eval.py --input "./Empathetic Dataset/selected_reply_cmnt/rjqlgc_ec_cache.json" \
                       --judge deepseek-v4

Batch mode  (input is a folder — all *.json files are evaluated):
    python run_eval.py --input "./Empathetic Dataset/selected_reply_cmnt" \
                       --judge deepseek-v4 --concurrency 10

Output is written to ./LLM_Judge/LLM_Judge_<judge>_<stem>.jsonl
"""

import argparse
import asyncio
import json
import logging
import statistics
from pathlib import Path

from llm_judge.evaluator import JUDGE_REGISTRY, get_output_path, load_json_samples, run_evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_SCORE_FIELDS = ["empathy", "relevance", "fluency", "overall_score"]
_OUTPUT_DIR = Path("LLM_AS_Judge")


def _print_summary(output_path: Path, label: str = "") -> None:
    records: list[dict] = []
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    valid = [r for r in records if r.get("overall_score", -1) != -1.0]
    failed = len(records) - len(valid)

    sep = "=" * 58
    title = f"G-EVAL SUMMARY  {label}".strip()
    print(f"\n{sep}")
    print(f"{title:^58}")
    print(sep)
    print(f"  {'Total evaluated:':<26} {len(records)}")
    print(f"  {'Valid / Failed:':<26} {len(valid)} / {failed}")
    print(f"  {'-' * 54}")
    print(f"  {'Metric':<22} {'Mean':>10} {'Std Dev':>10}")
    print(f"  {'-' * 54}")
    for field in _SCORE_FIELDS:
        vals = [r[field] for r in valid if r.get(field) is not None]
        if vals:
            mean = statistics.mean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            print(f"  {field:<22} {mean:>10.3f} {std:>10.3f}")
    print(sep)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "LLM-as-a-Judge G-Eval evaluation for empathetic response datasets. "
            "Pass a .json file for single-file mode, or a folder for batch mode."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help=(
            "Path to a single .json file OR a directory containing .json files. "
            "Single-file mode is chosen when the path ends with '.json'; "
            "otherwise all *.json files in the directory are processed."
        ),
    )
    parser.add_argument(
        "--judge",
        required=True,
        choices=list(JUDGE_REGISTRY.keys()),
        help="Which judge model to use.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Max concurrent API requests (default: 10).",
    )
    return parser.parse_args()


def _resolve_input_files(input_path: Path) -> list[Path]:
    """Return list of .json files to process based on input path."""
    if input_path.suffix.lower() == ".json":
        if not input_path.is_file():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        return [input_path]
    else:
        if not input_path.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {input_path}")
        files = sorted(input_path.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"No .json files found in: {input_path}")
        return files


def main() -> None:
    import os

    args = _parse_args()
    input_path: Path = args.input.resolve()

    judge_config = JUDGE_REGISTRY[args.judge]
    api_key = os.environ.get(judge_config["env_key"])
    if not api_key:
        raise EnvironmentError(
            f"API key not set. Run: export {judge_config['env_key']}=<your_key>"
        )

    input_files = _resolve_input_files(input_path)
    mode = "single" if len(input_files) == 1 else "batch"
    logger.info("Mode: %s | Files to process: %d", mode, len(input_files))
    logger.info("Judge:       %s (%s)", args.judge, judge_config["model_name"])
    logger.info("Concurrency: %d", args.concurrency)
    logger.info("Output dir:  %s", _OUTPUT_DIR.resolve())

    for json_file in input_files:
        output_path = get_output_path(json_file, args.judge, _OUTPUT_DIR)

        # Validate the file has required fields before starting
        samples = load_json_samples(json_file)
        missing = [
            i for i, s in enumerate(samples)
            if "post_id" not in s or "postkey" not in s or "feedback" not in s
        ]
        if missing:
            logger.error(
                "Skipping %s: %d record(s) missing required fields (post_id/postkey/feedback).",
                json_file.name,
                len(missing),
            )
            continue

        logger.info("--- Input:  %s (%d samples)", json_file.name, len(samples))
        logger.info("--- Output: %s", output_path)

        asyncio.run(
            run_evaluation(
                input_path=json_file,
                output_path=output_path,
                judge_key=args.judge,
                concurrency=args.concurrency,
                api_key=api_key,
            )
        )

        _print_summary(output_path, label=json_file.stem)


if __name__ == "__main__":
    main()
