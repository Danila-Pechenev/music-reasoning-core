import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/generate_benchmark.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("generate_benchmark", SCRIPT_PATH)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
generate_benchmark = importlib.util.module_from_spec(SCRIPT_SPEC)
sys.modules[SCRIPT_SPEC.name] = generate_benchmark
SCRIPT_SPEC.loader.exec_module(generate_benchmark)


class _FakeTask:
    def __init__(self):
        self.counter = 0

    def generate_example(self, *, level, max_tokens, mode):
        del max_tokens
        self.counter += 1
        return SimpleNamespace(
            prompt=f"Prompt {mode} level {level} example {self.counter}",
            answer=f"answer-{self.counter}",
            metadata={
                "mode": mode,
                "answer_kind": "text",
                "cot": f"reasoning-{self.counter}",
            },
        )


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_generate_benchmark_writes_nested_hugging_face_configurations(tmp_path, monkeypatch):
    tasks = {}

    def fake_get_task(family):
        return tasks.setdefault(family, _FakeTask())

    output_dir = tmp_path / "benchmark"
    monkeypatch.setattr(generate_benchmark, "get_task", fake_get_task)
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_benchmark.py",
            "--levels",
            "0",
            "--config-sizes",
            "1",
            "2",
            "--default-config-size",
            "1",
            "--out-dir",
            str(output_dir),
            "--seed",
            "7",
        ],
    )

    generate_benchmark.main()

    small_rows = _read_jsonl(output_dir / "data/n1/easy.jsonl")
    large_rows = _read_jsonl(output_dir / "data/n2/easy.jsonl")
    large_rows_by_id = {row["id"]: row for row in large_rows}

    assert len(small_rows) == 16
    assert len(large_rows) == 32
    assert len(large_rows_by_id) == len(large_rows)
    assert all(large_rows_by_id[row["id"]] == row for row in small_rows)

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["default_config"] == "n1"
    assert summary["configurations"]["n1"]["total_examples"] == 16
    assert summary["configurations"]["n2"]["total_examples"] == 32

    dataset_card = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "license: mit" in dataset_card
    assert "task_categories:\n- question-answering" in dataset_card
    assert "- config_name: n1\n  default: true" in dataset_card
    assert "path: data/n2/easy.jsonl" in dataset_card
    assert '"n1",\n    revision="v0.4.0"' in dataset_card


def test_generate_benchmark_removes_stale_data_files(tmp_path, monkeypatch):
    output_dir = tmp_path / "benchmark"
    stale_file = output_dir / "data/easy.jsonl"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text("stale\n", encoding="utf-8")

    monkeypatch.setattr(generate_benchmark, "get_task", lambda family: _FakeTask())
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_benchmark.py",
            "--levels",
            "0",
            "--config-sizes",
            "1",
            "--default-config-size",
            "1",
            "--out-dir",
            str(output_dir),
        ],
    )

    generate_benchmark.main()

    assert not stale_file.exists()
    assert (output_dir / "data/n1/easy.jsonl").exists()
