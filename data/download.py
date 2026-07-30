"""Download raw meeting corpora into data/raw/.

Everything lands as-is; normalization to the intermediate utterance schema is a
separate step (normalize.py) so a re-render never needs a re-download.
"""
import json
import subprocess
import sys
from pathlib import Path

from datasets import load_dataset

RAW = Path(__file__).parent / "raw"
RAW.mkdir(parents=True, exist_ok=True)


def save(ds_dict, name: str) -> None:
    out = RAW / name
    out.mkdir(exist_ok=True)
    for split, ds in ds_dict.items():
        path = out / f"{split}.jsonl"
        if path.exists():
            print(f"skip {path} (exists)")
            continue
        ds.to_json(str(path), force_ascii=False)
        print(f"wrote {path} ({len(ds)} rows)")


def main() -> None:
    # QMSum: per-turn speakers, query-based + general summaries (AMI + ICSI + committees)
    qmsum = RAW / "qmsum"
    if not qmsum.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/Yale-LILY/QMSum", str(qmsum)],
            check=True,
        )
    n = len(list(qmsum.glob("data/ALL/*/*.json")))
    print(f"qmsum: {n} meeting files")

    save(load_dataset("huuuyeah/meetingbank"), "meetingbank")
    save(load_dataset("knkarthick/dialogsum"), "dialogsum")
    save(load_dataset("renhehuang/vcsum-meeting-summary"), "vcsum")


if __name__ == "__main__":
    sys.exit(main())
