import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from bench import load_samples
from compute_metrics import compute_all_metrics
from PIL import Image
from transformers.hf_argparser import HfArgumentParser

BASELINE_RE = re.compile(r"^(\d+)_baseline_(\d+)\.png$")
TAYLORSEER_RE = re.compile(r"^(\d+)_taylorseer_(\d+)\.png$")


@dataclass
class ReportArguments:
    """Arguments for computing quality metrics from `eval/bench.py` outputs.

    - output_dir (str): Directory containing the `{idx}_baseline_{i}.png` /
      `{idx}_taylorseer_{i}.png` pairs saved by `eval/bench.py`.
    - data_path (str): The same --data_path used to run the benchmark, to recover
      each sample's prompt.
    - device (str): Device to run the metrics on.
    - report_path (str | None): Where to write the JSON report. Defaults to
      "<output_dir>/metrics.json".
    """

    output_dir: str = field(default="outputs")
    data_path: str = field(default="data/drawbench.jsonl")
    device: str = field(default="cuda")
    report_path: str | None = field(default=None)


def _index_images(output_dir: Path, pattern: re.Pattern) -> dict[tuple[int, int], Path]:
    images: dict[tuple[int, int], Path] = {}
    for path in output_dir.glob("*.png"):
        match = pattern.match(path.name)
        if match:
            images[(int(match.group(1)), int(match.group(2)))] = path
    return images


def collect_pairs(output_dir: str) -> list[tuple[int, Path, Path]]:
    """Pair up baseline/taylorseer images saved by `eval/bench.py` in `output_dir`."""
    root = Path(output_dir)
    baseline = _index_images(root, BASELINE_RE)
    taylorseer = _index_images(root, TAYLORSEER_RE)

    keys = sorted(baseline.keys() & taylorseer.keys())
    unpaired = baseline.keys() ^ taylorseer.keys()
    if unpaired:
        print(
            f"[report] skipping {len(unpaired)} unpaired image(s): {sorted(unpaired)}"
        )
    if not keys:
        raise RuntimeError(
            f"No matching baseline/taylorseer image pairs found in {output_dir}"
        )

    return [
        (global_idx, baseline[(global_idx, i)], taylorseer[(global_idx, i)])
        for global_idx, i in keys
    ]


def main(args: ReportArguments) -> None:
    samples = load_samples(args.data_path)
    pairs = collect_pairs(args.output_dir)

    target_images = [Image.open(baseline_path) for _, baseline_path, _ in pairs]
    pred_images = [Image.open(taylorseer_path) for _, _, taylorseer_path in pairs]
    prompts = [samples[global_idx].prompt for global_idx, _, _ in pairs]

    print(
        f"[report] computing metrics over {len(pairs)} paired sample(s) from {args.output_dir}"
    )
    metrics = compute_all_metrics(
        pred_images, target_images, prompts, device=args.device
    )

    report_path = Path(args.report_path or Path(args.output_dir) / "metrics.json")
    report_path.write_text(json.dumps(metrics, indent=2))
    print(f"[report] wrote {report_path}")


if __name__ == "__main__":
    parser = HfArgumentParser((ReportArguments,))
    (report_args,) = parser.parse_args_into_dataclasses(return_remaining_strings=False)
    main(report_args)
