#!/usr/bin/env python3
"""Generate a balanced music reasoning benchmark for Hugging Face upload.

The default output contains four nested Hugging Face configurations with 16,
32, 64, and 128 examples per mode. Every configuration preserves the easy,
moderate, and hard difficulty splits.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import shutil
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

from music_reasoning_tasks import __version__ as GENERATOR_VERSION
from music_reasoning_tasks import get_task
from music_reasoning_tasks.chord_roman_reasoning import MODE_NAMES as CHORD_ROMAN_MODES
from music_reasoning_tasks.pitch_interval_reasoning import MODE_NAMES as PITCH_INTERVAL_MODES


TASK_FAMILIES = {
    "pitch_interval_reasoning": tuple(PITCH_INTERVAL_MODES),
    "chord_roman_reasoning": tuple(CHORD_ROMAN_MODES),
}
DEFAULT_LEVELS = (0, 3, 5)
DEFAULT_SPLIT_NAMES = {
    0: "easy",
    3: "moderate",
    5: "hard",
}
DEFAULT_CONFIG_SIZES = (16, 32, 64, 128)
DEFAULT_CONFIG_SIZE = 64
DEFAULT_BENCHMARK_VERSION = "v0.4.2"
BENCHMARK_RESPONSE_INSTRUCTION = "Return only the requested answer."


def _jsonable(value: Any) -> Any:
    """Convert generated metadata into JSON-safe values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _metadata_json(metadata: Any) -> str:
    """Serialize metadata as a stable JSON string for simple HF schemas."""
    return json.dumps(_jsonable(dict(metadata)), ensure_ascii=False, sort_keys=True)


def _split_name(level: int) -> str:
    return DEFAULT_SPLIT_NAMES.get(level, f"level_{level}")


def _config_name(examples_per_mode: int) -> str:
    return f"n{examples_per_mode}"


def _benchmark_prompt(prompt: object) -> str:
    """Add the output-only instruction used for benchmark evaluation."""
    return f"{str(prompt).rstrip()} {BENCHMARK_RESPONSE_INSTRUCTION}"


def _row_from_example(example: Any, *, split: str, level: int, family: str, index: int) -> dict[str, Any]:
    """Create one benchmark row from a generated example."""
    metadata = dict(example.metadata)
    mode = str(metadata.get("mode", ""))
    answer_kind = str(metadata.get("answer_kind", "text"))
    cot = str(metadata.get("cot", ""))
    for redundant_key in ("prompt", "cot", "_time"):
        metadata.pop(redundant_key, None)
    row_id = f"{split}-{family}-{mode}-{index:04d}"
    return {
        "id": row_id,
        "split": split,
        "level": level,
        "difficulty": split,
        "family": family,
        "mode": mode,
        "prompt": _benchmark_prompt(example.prompt),
        "answer": str(example.answer),
        "answer_kind": answer_kind,
        "cot": cot,
        "metadata": _metadata_json(metadata),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _generate_mode_rows(
    *,
    family: str,
    mode: str,
    level: int,
    split: str,
    n_examples: int,
    max_tokens: int,
    deduplicate_prompts: bool,
    seen_prompts: set[tuple[str, str]],
    max_attempts: int,
) -> list[dict[str, Any]]:
    """Generate exactly n_examples rows for one family/mode/level cell."""
    task = get_task(family)
    rows: list[dict[str, Any]] = []
    attempts = 0
    while len(rows) < n_examples:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Could not generate {n_examples} unique examples for "
                f"{family}/{mode} at level {level} after {max_attempts} attempts."
            )

        example = task.generate_example(level=level, max_tokens=max_tokens, mode=mode)
        prompt_key = (split, _benchmark_prompt(example.prompt))
        if deduplicate_prompts and prompt_key in seen_prompts:
            continue

        metadata_mode = example.metadata.get("mode")
        if metadata_mode != mode:
            raise RuntimeError(f"Expected mode {mode!r}, got {metadata_mode!r}.")

        seen_prompts.add(prompt_key)
        rows.append(
            _row_from_example(
                example,
                split=split,
                level=level,
                family=family,
                index=len(rows),
            )
        )
    return rows


def _summarize_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return compact counts for one generated split."""
    family_counts = Counter(row["family"] for row in rows)
    mode_counts: dict[str, Counter[str]] = defaultdict(Counter)
    answer_kind_counts = Counter(row["answer_kind"] for row in rows)
    for row in rows:
        mode_counts[row["family"]][row["mode"]] += 1
    return {
        "num_examples": len(rows),
        "families": dict(sorted(family_counts.items())),
        "modes": {
            family: dict(sorted(counter.items()))
            for family, counter in sorted(mode_counts.items())
        },
        "answer_kinds": dict(sorted(answer_kind_counts.items())),
    }


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_dataset_card(
    path: Path,
    *,
    levels: list[int],
    config_sizes: list[int],
    default_config_size: int,
    seed: int | None,
    benchmark_version: str,
    generator_version: str,
) -> None:
    """Write a Hugging Face dataset card with one subset per benchmark size."""
    config_blocks: list[str] = []
    for config_size in config_sizes:
        config_name = _config_name(config_size)
        lines = [f"- config_name: {config_name}"]
        if config_size == default_config_size:
            lines.append("  default: true")
        lines.append("  data_files:")
        for level in levels:
            split = _split_name(level)
            lines.extend(
                [
                    f"  - split: {split}",
                    f"    path: data/{config_name}/{split}.jsonl",
                ]
            )
        config_blocks.append("\n".join(lines))

    total_modes = sum(len(modes) for modes in TASK_FAMILIES.values())
    config_names = [_config_name(size) for size in config_sizes]
    configuration_rows = "\n".join(
        f"| `{_config_name(size)}`{' (default)' if size == default_config_size else ''} "
        f"| {size} | {size * total_modes} | {size * total_modes * len(levels)} |"
        for size in config_sizes
    )
    split_rows = "\n".join(
        f"| `{_split_name(level)}` | {level} |" for level in levels
    )
    path.write_text(
        f"""---
pretty_name: Procedural Music Reasoning Benchmark
license: mit
language:
- en
task_categories:
- question-answering
tags:
- music
- music-theory
- reasoning
- benchmark
configs:
{chr(10).join(config_blocks)}
---

# Procedural Music Reasoning Benchmark

This benchmark was generated with
[`Danila-Pechenev/procedural-music-reasoning`](https://github.com/Danila-Pechenev/procedural-music-reasoning).

Benchmark version: `{benchmark_version}`.
Generator version: `{generator_version}`.

It contains balanced examples from two implemented music reasoning task
families:

- `pitch_interval_reasoning`
- `chord_roman_reasoning`

## Configurations

| Configuration | Examples per mode | Examples per split | Total examples |
|---|---:|---:|---:|
{configuration_rows}

The configurations are deterministic nested subsets in ascending size order:
{" is contained in ".join(f"`{name}`" for name in config_names)}. This makes
results obtained at different benchmark sizes directly comparable.

## Splits

Every configuration contains the same difficulty splits:

| Split | Generator level |
|---|---:|
{split_rows}
"""
        + f"""
## Columns

- `id`: stable row identifier.
- `split`: split name.
- `level`: generator difficulty/distribution level.
- `difficulty`: human-readable difficulty name.
- `family`: task family.
- `mode`: task mode within the family.
- `prompt`: model input.
- `answer`: canonical expected answer.
- `answer_kind`: answer-normalization family.
- `cot`: generator-produced reasoning trace.
- `metadata`: JSON string with symbolic generation metadata.

## Evaluation Protocol

For benchmark evaluation, give the model the `prompt` only and compare its
answer with `answer` using the task scorer. The `cot` field is provided for
inspection, supervised training, and error analysis, but should not be included
in the model prompt during benchmark evaluation.

Every benchmark prompt ends with `{BENCHMARK_RESPONSE_INSTRUCTION}` This
benchmark-only instruction requests the short answer expected by the scorer;
it is not added to examples produced directly by the task generators.

Difficulty levels are distributional. A hard split may still contain some
simple examples, but harder musical features are sampled more often or from a
larger space.

## Versioning

Hugging Face dataset releases should be tagged with the benchmark version. To
load this exact release after upload, use:

```python
from datasets import load_dataset

dataset = load_dataset(
    "dpechenev/music-reasoning-benchmark",
    "{_config_name(default_config_size)}",
    revision="{benchmark_version}",
)
```

Replace `{_config_name(default_config_size)}` with any configuration listed
above to select a different benchmark size.

Generation seed: `{seed}`.
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=list(DEFAULT_LEVELS),
        help="Generator levels to produce. Defaults to 0 3 5.",
    )
    parser.add_argument(
        "--config-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_CONFIG_SIZES),
        help="Nested configuration sizes in examples per mode. Defaults to 16 32 64 128.",
    )
    parser.add_argument(
        "--default-config-size",
        type=int,
        default=DEFAULT_CONFIG_SIZE,
        help="Configuration size marked as the Hugging Face default. Defaults to 64.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("benchmark_data/music_reasoning_benchmark"),
        help="Output directory containing data/, README.md, and summary.json.",
    )
    parser.add_argument("--max-tokens", type=int, default=8192, help="Token budget per generated example.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed. Use -1 for no explicit seed.")
    parser.add_argument(
        "--benchmark-version",
        default=DEFAULT_BENCHMARK_VERSION,
        help=f"Benchmark release version recorded in README.md and summary.json. Defaults to {DEFAULT_BENCHMARK_VERSION}.",
    )
    parser.add_argument(
        "--generator-version",
        default=GENERATOR_VERSION,
        help="Generator package version recorded in README.md and summary.json.",
    )
    parser.add_argument(
        "--max-attempts-multiplier",
        type=int,
        default=80,
        help="Maximum attempts per cell equals the largest configuration size times this value.",
    )
    parser.add_argument(
        "--allow-duplicate-prompts",
        action="store_true",
        help="Allow duplicate prompts inside the same split.",
    )
    parser.add_argument(
        "--write-combined",
        action="store_true",
        help="Also write one data/<config>/all.jsonl file per configuration for local analysis.",
    )
    args = parser.parse_args()

    config_sizes = sorted(set(args.config_sizes))
    if not config_sizes or config_sizes[0] <= 0:
        parser.error("--config-sizes must contain positive integers.")
    if args.default_config_size not in config_sizes:
        parser.error("--default-config-size must be one of --config-sizes.")
    if args.max_attempts_multiplier <= 0:
        parser.error("--max-attempts-multiplier must be positive.")
    if len(set(args.levels)) != len(args.levels):
        parser.error("--levels must not contain duplicates.")

    seed = None if args.seed == -1 else args.seed
    if seed is not None:
        random.seed(seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.out_dir / "data"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "benchmark_version": args.benchmark_version,
        "generator_version": args.generator_version,
        "response_instruction": BENCHMARK_RESPONSE_INSTRUCTION,
        "levels": args.levels,
        "config_sizes": config_sizes,
        "default_config": _config_name(args.default_config_size),
        "seed": seed,
        "families": {family: list(modes) for family, modes in TASK_FAMILIES.items()},
        "configurations": {
            _config_name(size): {
                "examples_per_mode": size,
                "splits": {},
            }
            for size in config_sizes
        },
    }
    rows_by_config: dict[str, list[dict[str, Any]]] = {
        _config_name(size): [] for size in config_sizes
    }
    largest_config_size = config_sizes[-1]

    for level in args.levels:
        split = _split_name(level)
        split_rows_by_config: dict[str, list[dict[str, Any]]] = {
            _config_name(size): [] for size in config_sizes
        }
        seen_prompts: set[tuple[str, str]] = set()
        print(f"Generating split {split!r} at level {level}...")
        for family, modes in TASK_FAMILIES.items():
            for mode in modes:
                print(f"  {family}/{mode}: {largest_config_size}")
                rows = _generate_mode_rows(
                    family=family,
                    mode=mode,
                    level=level,
                    split=split,
                    n_examples=largest_config_size,
                    max_tokens=args.max_tokens,
                    deduplicate_prompts=not args.allow_duplicate_prompts,
                    seen_prompts=seen_prompts,
                    max_attempts=largest_config_size * args.max_attempts_multiplier,
                )
                for size in config_sizes:
                    split_rows_by_config[_config_name(size)].extend(rows[:size])

        for size in config_sizes:
            config_name = _config_name(size)
            split_rows = split_rows_by_config[config_name]
            _write_jsonl(data_dir / config_name / f"{split}.jsonl", split_rows)
            rows_by_config[config_name].extend(split_rows)
            summary["configurations"][config_name]["splits"][split] = _summarize_split(split_rows)

    for config_name, config_rows in rows_by_config.items():
        if args.write_combined:
            _write_jsonl(data_dir / config_name / "all.jsonl", config_rows)
        summary["configurations"][config_name]["total_examples"] = len(config_rows)

    _write_summary(args.out_dir / "summary.json", summary)
    _write_dataset_card(
        args.out_dir / "README.md",
        levels=args.levels,
        config_sizes=config_sizes,
        default_config_size=args.default_config_size,
        seed=seed,
        benchmark_version=args.benchmark_version,
        generator_version=args.generator_version,
    )
    print(f"Done. Wrote benchmark to {args.out_dir}")


if __name__ == "__main__":
    main()
