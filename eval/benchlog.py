"""Shared run-log format for `eval/bench.py` and `eval/compute_metrics.py`.

A run log is a JSONL file describing one benchmark run of a single method
(`base` or `taylorseer`). It holds two kinds of records, tagged by `"type"`:

- `meta`: written once per worker (one per data-parallel chunk). Records the
  method, the model / generation / TaylorSeer hyperparameters and the benchmark
  sharding, so a run is reproducible from its log alone.
- `sample`: one per generated sample. Records the global sample index, the text
  prompt, the generation seed, the per-sample latency and the image path.

`eval/compute_metrics.py` consumes two such logs (one `base`, one
`taylorseer`), pairs their samples by global index and computes the quality
metrics and the speedup from them.
"""

import json
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Any

META_RECORD = "meta"
SAMPLE_RECORD = "sample"


@dataclass
class Sample:
    """A single test sample, i.e. a text prompt plus the data file's extra fields."""

    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.prompt


def load_samples(data_path: str | Path) -> list[Sample]:
    """Load samples from a JSONL data file, one JSON object with a `prompt` per line."""
    samples: list[Sample] = []
    with open(data_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            prompt = data.pop("prompt")
            samples.append(Sample(prompt=prompt, metadata=data))
    return samples


class RunLogWriter:
    """Append-only JSONL writer for a single benchmark worker.

    The file is truncated on open (one writer owns one file) and flushed after
    every record, so a partially finished run still yields a readable log.
    """

    def __init__(self, path: str | Path, meta: dict[str, Any]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w")
        self._write(META_RECORD, meta)

    def _write(self, record_type: str, record: dict[str, Any]) -> None:
        self._file.write(json.dumps({"type": record_type, **record}) + "\n")
        self._file.flush()

    def write_sample(self, **fields: Any) -> None:
        self._write(SAMPLE_RECORD, fields)

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "RunLogWriter":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


@dataclass
class SampleRecord:
    """One `sample` record of a run log, with its image path already resolved."""

    index: int
    prompt: str
    seed: int
    latency: float
    image_path: Path
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunLog:
    """The parsed content of one or more run-log files of the *same* run."""

    paths: list[Path]
    metas: list[dict[str, Any]]
    samples: dict[int, SampleRecord]

    @property
    def method(self) -> str:
        return str(self.metas[0].get("method", "unknown")) if self.metas else "unknown"

    @property
    def run_name(self) -> str:
        return (
            str(self.metas[0].get("run_name", self.method))
            if self.metas
            else self.method
        )

    @property
    def config(self) -> dict[str, Any]:
        """Run configuration from the first `meta` record, without sharding fields."""
        if not self.metas:
            return {}
        config = {k: v for k, v in self.metas[0].items() if k != "type"}
        benchmark = config.get("benchmark")
        if isinstance(benchmark, dict):
            config["benchmark"] = {
                k: v
                for k, v in benchmark.items()
                if k not in ("chunk_idx", "num_chunks", "num_chunk_samples")
            }
        return config

    @property
    def data_path(self) -> str | None:
        benchmark = self.metas[0].get("benchmark", {}) if self.metas else {}
        data_path = benchmark.get("data_path") if isinstance(benchmark, dict) else None
        return str(data_path) if data_path else None


def _resolve_image_path(raw_path: str, log_path: Path) -> Path:
    """Resolve a logged image path, falling back to paths relative to the log file."""
    path = Path(raw_path)
    if path.exists():
        return path
    for base in (log_path.parent, log_path.parent.parent):
        candidate = base / path
        if candidate.exists():
            return candidate
    return path


def expand_log_paths(spec: str) -> list[Path]:
    """Expand a comma-separated list of log paths / glob patterns."""
    paths: list[Path] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        matches = sorted(glob(part))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(part))
    return paths


def load_run_log(spec: str) -> RunLog:
    """Load a run log from a path, a glob pattern or a comma-separated list of both.

    Passing the per-chunk logs of one run (e.g. `outputs/logs/base.chunk*.jsonl`)
    is equivalent to passing their merged file: every `meta` record is kept and
    the `sample` records are indexed by their global sample index.
    """
    paths = expand_log_paths(spec)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"run log(s) not found: {', '.join(missing)}")
    if not paths:
        raise FileNotFoundError(f"no run log matched {spec!r}")

    metas: list[dict[str, Any]] = []
    samples: dict[int, SampleRecord] = {}
    duplicates: list[int] = []
    for path in paths:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record_type = record.get("type")
                if record_type == META_RECORD:
                    metas.append(record)
                elif record_type == SAMPLE_RECORD:
                    index = int(record["index"])
                    if index in samples:
                        duplicates.append(index)
                        continue
                    samples[index] = SampleRecord(
                        index=index,
                        prompt=record["prompt"],
                        seed=int(record["seed"]),
                        latency=float(record["latency"]),
                        image_path=_resolve_image_path(record["image_path"], path),
                        raw=record,
                    )
    if duplicates:
        print(
            f"[log] {spec}: ignored {len(duplicates)} duplicate sample record(s): "
            f"{sorted(set(duplicates))}"
        )

    methods = {meta.get("method") for meta in metas}
    if len(methods) > 1:
        raise ValueError(
            f"run log {spec!r} mixes several methods: {sorted(map(str, methods))}"
        )

    return RunLog(paths=paths, metas=metas, samples=samples)
