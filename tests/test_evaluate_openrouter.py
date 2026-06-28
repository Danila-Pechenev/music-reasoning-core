import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/evaluate_openrouter.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("evaluate_openrouter", SCRIPT_PATH)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
evaluate_openrouter = importlib.util.module_from_spec(SCRIPT_SPEC)
sys.modules[SCRIPT_SPEC.name] = evaluate_openrouter
SCRIPT_SPEC.loader.exec_module(evaluate_openrouter)


def _benchmark_row():
    return {
        "id": "easy-family-mode-0000",
        "split": "easy",
        "family": "family",
        "mode": "mode",
        "prompt": "Prompt",
        "answer": "Answer",
        "metadata": "{}",
    }


@pytest.mark.parametrize(
    ("extra_args", "expected_config"),
    [
        ([], "n64"),
        (["--dataset-config", "n16"], "n16"),
    ],
)
def test_parse_args_selects_dataset_configuration(monkeypatch, extra_args, expected_config):
    monkeypatch.setattr("sys.argv", ["evaluate_openrouter.py", "provider/model", *extra_args])

    args = evaluate_openrouter._parse_args()

    assert args.dataset_config == expected_config
    assert args.provider is None
    assert args.batch_size == 256
    assert args.seed == 0
    assert not hasattr(args, "concurrent_batches")


def test_parse_args_accepts_openrouter_provider_slug(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate_openrouter.py", "deepseek/deepseek-v4-flash", "--provider", "baidu"],
    )

    args = evaluate_openrouter._parse_args()

    assert args.provider == "baidu"


def test_result_directory_separates_max_token_budgets(tmp_path):
    result_dir = evaluate_openrouter._result_directory(
        tmp_path,
        "openrouter/deepseek/deepseek-v4-flash",
        "v0.4.2",
        "n16",
        "baidu",
        "high",
        8192,
        0,
    )

    assert result_dir == (
        tmp_path
        / "deepseek-deepseek-v4-flash"
        / "v0.4.2"
        / "n16"
        / "provider-baidu"
        / "reasoning-high"
        / "max-tokens-8192"
        / "seed-0"
    )


def test_cli_help_describes_every_optional_argument_default(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["evaluate_openrouter.py", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        evaluate_openrouter._parse_args()

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    normalized_help = " ".join(help_text.split())
    assert normalized_help.count("Defaults to") == 16
    assert "Required exact OpenRouter model ID" in normalized_help


def test_load_benchmark_pins_configuration_to_exact_commit(monkeypatch):
    calls = []
    commit = "a" * 40

    def fake_load_dataset(repo, config, *, revision):
        calls.append((repo, config, revision))
        return {"easy": [_benchmark_row()]}

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))

    rows = evaluate_openrouter._load_benchmark(
        "owner/benchmark",
        "n16",
        commit,
        ["easy"],
        None,
    )

    assert calls == [("owner/benchmark", "n16", commit)]
    assert rows == [_benchmark_row()]


def test_resume_requires_matching_dataset_configuration():
    row = _benchmark_row()
    result = {
        "status": "ok",
        "model_requested": "openrouter/provider/model",
        "provider_requested": None,
        "dataset_repo": "owner/benchmark",
        "dataset_config": "n64",
        "dataset_commit": "abc123",
        "reasoning_effort": None,
        "temperature": 0.0,
        "max_tokens": 1024,
        "seed": 0,
        "prompt_sha256": evaluate_openrouter._prompt_hash(row["prompt"]),
        "expected_answer": row["answer"],
    }

    assert evaluate_openrouter._is_reusable_result(
        result,
        row,
        "openrouter/provider/model",
        None,
        "owner/benchmark",
        "n64",
        "abc123",
        None,
        0.0,
        1024,
        0,
    )
    assert not evaluate_openrouter._is_reusable_result(
        result,
        row,
        "openrouter/provider/model",
        None,
        "owner/benchmark",
        "n16",
        "abc123",
        None,
        0.0,
        1024,
        0,
    )
    assert not evaluate_openrouter._is_reusable_result(
        result,
        row,
        "openrouter/provider/model",
        "baidu",
        "owner/benchmark",
        "n64",
        "abc123",
        None,
        0.0,
        1024,
        0,
    )


def test_call_rows_sends_openrouter_reasoning_object(monkeypatch):
    captured_kwargs = {}

    class FakeResponse:
        usage = {}
        model_used = "openrouter/provider/model"
        reasoning = None

        def __str__(self):
            return "Answer"

    def fake_complete(inputs, **kwargs):
        captured_kwargs.update(kwargs)
        return [FakeResponse() for _ in inputs]

    monkeypatch.setitem(sys.modules, "litlm", SimpleNamespace(complete=fake_complete))
    monkeypatch.setattr(evaluate_openrouter, "score_answer", lambda answer, row: 1.0)

    results = evaluate_openrouter._call_rows(
        [_benchmark_row()],
        model="openrouter/provider/model",
        provider="baidu",
        max_tokens=8192,
        temperature=0.0,
        num_retries=5,
        timeout=120.0,
        caching=False,
        reasoning_effort="high",
        seed=0,
    )

    assert captured_kwargs["reasoning"] == {"effort": "high"}
    assert captured_kwargs["extra_body"] == {"provider": {"only": ["baidu"]}}
    assert captured_kwargs["seed"] == 0
    assert "reasoning_effort" not in captured_kwargs
    assert results[0]["status"] == "ok"


def test_call_batches_runs_batches_sequentially(monkeypatch):
    call_order = []

    def fake_call_rows(rows, **kwargs):
        del kwargs
        call_order.append([row["id"] for row in rows])
        return [{"id": row["id"]} for row in rows]

    rows = [{"id": str(index)} for index in range(8)]
    monkeypatch.setattr(evaluate_openrouter, "_call_rows", fake_call_rows)

    completed_batches = list(
        evaluate_openrouter._call_batches(
            rows,
            batch_size=2,
            model="openrouter/provider/model",
            provider=None,
            max_tokens=1024,
            temperature=0.0,
            num_retries=1,
            timeout=10.0,
            caching=False,
            reasoning_effort=None,
            seed=0,
        )
    )

    assert call_order == [["0", "1"], ["2", "3"], ["4", "5"], ["6", "7"]]
    assert sorted(row["id"] for batch in completed_batches for row in batch) == [
        str(index) for index in range(8)
    ]


def test_recorded_benchmark_time_merges_overlapping_batch_intervals():
    results = [
        {
            "batch_id": "first",
            "batch_seconds": 10.0,
            "batch_started_at": 100.0,
            "batch_finished_at": 110.0,
        },
        {
            "batch_id": "second",
            "batch_seconds": 10.0,
            "batch_started_at": 105.0,
            "batch_finished_at": 115.0,
        },
        {
            "batch_id": "third",
            "batch_seconds": 5.0,
            "batch_started_at": 120.0,
            "batch_finished_at": 125.0,
        },
    ]

    assert evaluate_openrouter._recorded_benchmark_seconds(results) == 20.0


def test_report_separates_aggregate_metrics_from_incorrect_responses(tmp_path):
    base_result = {
        "split": "easy",
        "level": 0,
        "family": "pitch_interval_reasoning",
        "mode": "interval_naming",
        "status": "ok",
        "score": 1.0,
        "usage": {},
        "cost": 0.0,
    }
    correct = {
        **base_result,
        "id": "right-id",
        "prompt": "Correct prompt",
        "expected_answer": "major third",
        "prediction": "major third",
    }
    incorrect = {
        **base_result,
        "id": "wrong-id",
        "prompt": "Incorrect prompt",
        "expected_answer": "minor second",
        "prediction": "major second",
        "score": 0.0,
    }
    api_error = {
        **base_result,
        "id": "api-error-id",
        "prompt": "Failed prompt",
        "expected_answer": "perfect fifth",
        "prediction": "",
        "status": "api_error",
        "score": 0.0,
        "error": "Provider failed",
    }
    results = [correct, incorrect, api_error]
    report_path = tmp_path / "report.md"
    incorrect_path = tmp_path / "incorrect_responses.md"

    evaluate_openrouter._write_report(
        report_path,
        splits=["easy"],
        dataset_repo="owner/benchmark",
        dataset_config="n16",
        dataset_revision="v0.3.0",
        dataset_commit="abc123",
        generator_version="0.3.0",
        model="openrouter/provider/model",
        provider="baidu",
        reasoning_effort=None,
        results=results,
        split_benchmark_seconds={"easy": 1.0},
        temperature=0.0,
        max_tokens=1024,
        seed=0,
    )
    evaluate_openrouter._write_incorrect_responses(
        incorrect_path,
        splits=["easy"],
        dataset_repo="owner/benchmark",
        dataset_config="n16",
        dataset_revision="v0.3.0",
        model="openrouter/provider/model",
        provider="baidu",
        reasoning_effort=None,
        seed=0,
        results=results,
    )

    report = report_path.read_text(encoding="utf-8")
    accuracy_note = "`Accuracy` uses all requested rows in each split as the denominator."
    assert report.index("## Results by Difficulty") < report.index(accuracy_note)
    assert report.index(accuracy_note) < report.index("## Easy Split")
    assert "Incorrect prompt" not in report
    assert "Provider failed" not in report
    assert "Sample Incorrect Responses" not in report
    assert "Provider routing:** `baidu`" in report
    assert "API seed:** `0`" in report

    incorrect_report = incorrect_path.read_text(encoding="utf-8")
    assert "wrong-id" in incorrect_report
    assert "Incorrect prompt" in incorrect_report
    assert "minor second" in incorrect_report
    assert "major second" in incorrect_report
    assert "right-id" not in incorrect_report
    assert "api-error-id" not in incorrect_report
    assert "Provider routing:** `baidu`" in incorrect_report
    assert "API seed:** `0`" in incorrect_report
