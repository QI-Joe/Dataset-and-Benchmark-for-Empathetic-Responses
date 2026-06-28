import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from openai import AsyncOpenAI

from .prompts import ACADEMIC_GEVAL_SYSTEM_PROMPT, build_user_message, extract_json

logger = logging.getLogger(__name__)

JUDGE_REGISTRY: dict[str, dict[str, str]] = {
    "glm-5.2": {
        "base_url": "https://ws-oc36xob6rqt0o3b3.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
        "model_name": "glm-5.2",
        "env_key": "DASHSCOPE_API_KEY",
    },
    "qwen3.7-max": {
        "base_url": "https://ws-oc36xob6rqt0o3b3.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
        "model_name": "qwen-max",
        "env_key": "DASHSCOPE_API_KEY",
    },
    "grok-beta": {
        "base_url": "https://api.x.ai/v1",
        "model_name": "grok-beta",
        "env_key": "XAI_API_KEY",
    },
}


def load_json_samples(path: Path) -> list[dict]:
    """Load samples from a JSON array file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(data).__name__}")
    return data


def get_output_path(input_path: Path, judge_key: str, output_dir: Path) -> Path:
    """Derive the output JSONL path for a given input file under output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"LLM_Judge_{judge_key}_{input_path.stem}.jsonl"


async def _evaluate_sample(
    client: AsyncOpenAI,
    model_name: str,
    judge_key: str,
    sample: dict,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
) -> dict:
    """Evaluate one sample; returns a dict preserving all original fields plus scores."""
    result: dict = {
        "post_id": sample["post_id"],
        "postkey": sample["postkey"],
        "feedback": sample["feedback"],
        "response": sample.get("response"),
        "unknow_len_idx": sample.get("unknow_len_idx"),
        "judge_model": judge_key,
        "empathy": -1,
        "relevance": -1,
        "fluency": -1,
        "overall_score": -1.0,
        "justification": "",
        "error": None,
    }

    user_msg = build_user_message(sample["postkey"], sample["feedback"], sample["response"])

    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": ACADEMIC_GEVAL_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.0,
                )
                parsed = extract_json(response.choices[0].message.content)
                result["empathy"] = int(parsed["empathy"])
                result["relevance"] = int(parsed["relevance"])
                result["fluency"] = int(parsed["fluency"])
                result["overall_score"] = float(parsed["overall_score"])
                result["justification"] = parsed.get("justification", "")
                # logger.info(
                #     "Evaluated post_id %s | Overall: %.1f | Empathy: %d",
                #     result["post_id"],
                #     result["overall_score"],
                #     result["empathy"],
                # )
                return result
            except Exception as exc:
                logger.warning(
                    "Attempt %d/%d failed for post_id %s: %s",
                    attempt + 1,
                    max_retries,
                    sample["post_id"],
                    exc,
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(2**attempt)

    result["error"] = "max_retries_exceeded"
    logger.error("Failed post_id %s after %d attempts", sample["post_id"], max_retries)
    return result


async def run_evaluation(
    input_path: Path,
    output_path: Path,
    judge_key: str,
    concurrency: int,
    api_key: str,
) -> None:
    config = JUDGE_REGISTRY[judge_key]
    client = AsyncOpenAI(base_url=config["base_url"], api_key=api_key)

    # Resume support: track already-processed post_ids
    processed_ids: set[int] = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    processed_ids.add(json.loads(line)["post_id"])
        logger.info("Resuming: %d samples already processed.", len(processed_ids))

    samples = load_json_samples(input_path)
    pending = [s for s in samples if s["post_id"] not in processed_ids]
    logger.info("Total: %d | Pending: %d", len(samples), len(pending))

    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        _evaluate_sample(client, config["model_name"], judge_key, s, semaphore)
        for s in pending
    ]

    with open(output_path, "a", encoding="utf-8") as out_f:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()
