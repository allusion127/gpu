#!/usr/bin/env python3
"""Self-contained contracts for run_screen_exact_campaign.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "run_screen_exact_campaign.py"


def load_module():
    spec = importlib.util.spec_from_file_location("screen_exact", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_fake_rasbery(path: Path) -> None:
    path.write_text(
        r'''#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
ri = args.index("--rasi")
ro = args.index("--raso")
bm = args.index("--batch-mode")
inputs = [Path(v) for v in args[ri + 1:ro]]
outputs = [Path(v) for v in args[ro + 1:bm]]
if len(inputs) != len(outputs) or int(args[bm + 1]) != len(inputs):
    raise SystemExit(91)
is_screen = os.environ.get("RASBERY_BATCH_LIGHT_RESULT") == "1"
for inp, out in zip(inputs, outputs):
    payload = json.loads(inp.read_text(encoding="utf-8"))
    out.parent.mkdir(parents=True, exist_ok=True)
    if is_screen:
        if "RASBERY_GA_FEEDBACK_PASSES" not in os.environ:
            raise SystemExit(92)
        out.write_text(json.dumps({
            "candidate_id": payload["id"],
            "objective": {"score": payload["score"]},
            "physics_mode": "ga_screen_feedback_limited",
            "feedback_passes": int(os.environ["RASBERY_GA_FEEDBACK_PASSES"]),
            "requires_exact_rerun": True
        }) + "\n", encoding="utf-8")
    else:
        if os.environ.get("FAKE_EXACT_FAIL") == payload["id"]:
            raise SystemExit(93)
        out.write_bytes(("EXACT:" + payload["id"]).encode("utf-8"))
''',
        encoding="utf-8",
    )
    path.chmod(0o755)


def make_manifest(root: Path, count: int = 4) -> Path:
    rows = []
    for index in range(count):
        candidate_id = f"c{index}"
        input_path = root / f"{candidate_id}.json"
        input_path.write_text(
            json.dumps({"id": candidate_id, "score": float(index)}),
            encoding="utf-8",
        )
        rows.append({"id": candidate_id, "input": input_path.name})
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return manifest


def run_campaign(root: Path, extra: list[str] | None = None, env=None):
    fake = root / "fake_rasbery.py"
    write_fake_rasbery(fake)
    manifest = make_manifest(root)
    workdir = root / "campaign"
    command = [
        sys.executable,
        str(TOOL),
        "--manifest",
        str(manifest),
        "--workdir",
        str(workdir),
        "--executable",
        str(fake),
        "--batch-width",
        "2",
        "--screen-feedback-passes",
        "2",
        "--survivors",
        "2",
        "--score-key",
        "objective.score",
        "--maximize",
        "--master-w16-cases-per-hour",
        "0.01",
        "--target-speedup",
        "1.0",
    ]
    if extra:
        command.extend(extra)
    return subprocess.run(command, text=True, capture_output=True, env=env), workdir


def test_pure_contracts(module) -> None:
    candidates = [
        module.Candidate(f"c{i}", Path(f"i{i}"), Path(f"s{i}"), Path(f"e{i}"))
        for i in range(5)
    ]
    assert [len(chunk) for chunk in module.chunked(candidates, 2)] == [2, 2, 1]
    try:
        list(module.chunked(candidates, 65))
    except module.CampaignError:
        pass
    else:
        raise AssertionError("batch width >64 was accepted")
    assert module.effective_throughput(64, 3600.0) == 64.0
    assert module.speedup_vs_master(4360.0, 218.0) == 20.0


def test_happy_path() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        completed, workdir = run_campaign(Path(tmp))
        assert completed.returncode == 0, completed.stdout + completed.stderr
        receipt = json.loads((workdir / "campaign_receipt.json").read_text())
        assert receipt["valid"] is True
        assert receipt["requires_exact_rerun"] is False
        assert receipt["screen"]["approximate"] is True
        assert receipt["screen"]["physics_mode"] == "ga_screen_feedback_limited"
        assert receipt["exact"]["approximate"] is False
        assert len(receipt["exact"]["outputs"]) == 2
        ranking = json.loads((workdir / "screen_ranking.json").read_text())
        selected = [row["id"] for row in ranking if row["selected_for_exact"]]
        assert selected == ["c3", "c2"], selected
        assert all((workdir / "exact" / f"{cid}.h5").is_file() for cid in selected)


def test_below_target_is_valid_but_fails_gate() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        completed, workdir = run_campaign(
            Path(tmp),
            ["--master-w16-cases-per-hour", "1e12", "--target-speedup", "20"],
        )
        assert completed.returncode == 3, completed.stdout + completed.stderr
        receipt = json.loads((workdir / "campaign_receipt.json").read_text())
        assert receipt["valid"] is True
        assert receipt["requires_exact_rerun"] is False
        assert receipt["target_met"] is False


def test_duplicate_input_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        fake = root / "fake_rasbery.py"
        write_fake_rasbery(fake)
        input_path = root / "same.json"
        input_path.write_text(json.dumps({"id": "same", "score": 1.0}))
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps([
                {"id": "a", "input": "same.json"},
                {"id": "b", "input": "same.json"},
            ]),
            encoding="utf-8",
        )
        workdir = root / "campaign"
        completed = subprocess.run(
            [
                sys.executable,
                str(TOOL),
                "--manifest", str(manifest),
                "--workdir", str(workdir),
                "--executable", str(fake),
                "--survivors", "1",
                "--score-key", "objective.score",
            ],
            text=True,
            capture_output=True,
        )
        assert completed.returncode == 2
        receipt = json.loads((workdir / "campaign_receipt.json").read_text())
        assert receipt["valid"] is False
        assert receipt["requires_exact_rerun"] is True


def main() -> int:
    module = load_module()
    test_pure_contracts(module)
    test_happy_path()
    test_below_target_is_valid_but_fails_gate()
    test_duplicate_input_fails_closed()
    print("screen-exact campaign contracts: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
