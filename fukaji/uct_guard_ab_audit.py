"""Independent, read-only audit of a same-run ABBA UCT reserve experiment.

The two pinned engine revisions differ only in the UCT reserve and its tests.
No model training, games, or promotion. Each revision runs the existing
SHA-checked 12-position probe twice, in baseline/guard/guard/baseline order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path

SOURCE_SHA = "60eb9ab4dadc3776e43ac49cf27bd04137477c339850fdaa7db2265248a8d3bf"
MODEL_SHA = "18b39f647caa458389a1900b875337adadac74ae37e27343e7f0094ed663c492"
BASE_REF = "b2001063bac5f84d4ee28004616a599c8c3da574"
GUARD_REF = "378bd6426a193eac8d9a936c3f4252242b8707f4"
REQUEST = {
    "enabled": False, "research_only": True, "model_adoption_allowed": False,
    "source_run": 40637486947, "g8_model_run": 40636602223,
    "baseline_ref": BASE_REF, "guard_ref": GUARD_REF,
    "positions": 12, "movetime_ms": 200, "repeats_per_variant": 2,
    "runner_jobs": 1,
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate_request(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value not in (REQUEST, dict(REQUEST, enabled=True)):
        raise ValueError("unapproved UCT guard ABBA request")
    enabled = value["enabled"]
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as stream:
            print("enabled=" + str(enabled).lower(), file=stream)
    return enabled


def load_run(folder):
    root = Path(folder)
    raw = (root / "position_diagnostics.jsonl").read_bytes()
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
    if (summary.get("status") != "read_only_search_diagnostic_not_strength_or_adoption"
            or summary.get("diagnostics_sha256") != digest(raw)
            or summary.get("source_games_sha256") != SOURCE_SHA
            or summary.get("g8_model_sha256") != MODEL_SHA
            or summary.get("source_run") != REQUEST["source_run"]
            or summary.get("g8_model_run") != REQUEST["g8_model_run"]
            or summary.get("positions") != 12 or len(rows) != 12
            or summary.get("movetime_ms_each") != 200
            or summary.get("unresolved_source_games_used_as_teacher") is not False
            or summary.get("model_promoted") is not False
            or summary.get("official_kumoji_changed") is not False):
        raise ValueError("source, SHA or experiment summary mismatch")
    keys = [(r["pair"], r["target_ply"], r["sfen"]) for r in rows]
    if (len(set(keys)) != 12 or len({r["pair"] for r in rows}) != 12
            or sorted(r["target_ply"] for r in rows) != [12]*4 + [50]*4 + [110]*4):
        raise ValueError("non-independent or changed 12-position sample")
    for row in rows:
        m = row["methods"]["uct"]
        if (not isinstance(m["elapsed_ms"], (float, int))
                or m["elapsed_ms"] <= 0 or m["iterations"] < 1
                or not m["move"] or row["in_check"]):
            raise ValueError("invalid nonterminal UCT observation")
    return rows


def audit(base_a, guard_a, guard_b, base_b, output):
    runs = [load_run(p) for p in (base_a, guard_a, guard_b, base_b)]
    output_rows = []
    for index in range(12):
        records = [run[index] for run in runs]
        key = ("pair", "target_ply", "sfen", "hand_pieces", "g8_eval_cp")
        if any(tuple(r[k] for k in key) != tuple(records[0][k] for k in key)
               for r in records[1:]):
            raise ValueError("ABBA position or frozen evaluation mismatch")
        old = [records[0]["methods"]["uct"], records[3]["methods"]["uct"]]
        new = [records[1]["methods"]["uct"], records[2]["methods"]["uct"]]
        old_ms = [r["elapsed_ms"] for r in old]
        new_ms = [r["elapsed_ms"] for r in new]
        output_rows.append({
            "pair": records[0]["pair"], "target_ply": records[0]["target_ply"],
            "sfen": records[0]["sfen"], "hand_pieces": records[0]["hand_pieces"],
            "baseline_ms": old_ms, "guard_ms": new_ms,
            "baseline_iterations": [r["iterations"] for r in old],
            "guard_iterations": [r["iterations"] for r in new],
            "baseline_moves": [r["move"] for r in old],
            "guard_moves": [r["move"] for r in new],
            "paired_median_delta_ms": round(statistics.median(new_ms)-statistics.median(old_ms), 3),
        })
    base_times = [ms for r in output_rows for ms in r["baseline_ms"]]
    guard_times = [ms for r in output_rows for ms in r["guard_ms"]]
    summary = {
        "status": "research_only_same_12_positions_abba_no_strength_claim",
        "source_games_sha256": SOURCE_SHA, "g8_model_sha256": MODEL_SHA,
        "baseline_ref": BASE_REF, "guard_ref": GUARD_REF,
        "run_order": ["baseline_a", "guard_a", "guard_b", "baseline_b"],
        "positions": 12, "observations_per_variant": 24,
        "movetime_ms_each": 200,
        "baseline_over_200ms": sum(x > 200 for x in base_times),
        "guard_over_200ms": sum(x > 200 for x in guard_times),
        "baseline_median_ms": round(statistics.median(base_times), 3),
        "guard_median_ms": round(statistics.median(guard_times), 3),
        "guard_faster_positions": sum(r["paired_median_delta_ms"] < 0 for r in output_rows),
        "baseline_total_iterations": sum(sum(r["baseline_iterations"]) for r in output_rows),
        "guard_total_iterations": sum(sum(r["guard_iterations"]) for r in output_rows),
        "limitations": [
            "Same runner and ABBA order reduce but do not eliminate OS scheduling variation.",
            "Two repeated observations per variant are descriptive, not a significance test.",
            "The 12 positions are selected, not a representative random sample.",
            "The guard cannot interrupt a slow iteration; strict 200ms compliance is unproven.",
            "Move disagreement and iteration counts do not establish playing strength.",
            "No new games, teacher labels, model training, or official adoption.",
        ],
    }
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False)+"\n" for r in output_rows)
    (root / "abba_positions.jsonl").write_text(data, encoding="utf-8")
    summary["abba_positions_sha256"] = digest(data.encode("utf-8"))
    (root / "abba_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False)+"\n",
        encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-request")
    for name in ("base-a", "guard-a", "guard-b", "base-b", "out"):
        parser.add_argument("--" + name)
    args = parser.parse_args()
    if args.check_request:
        print(json.dumps({"enabled": validate_request(args.check_request)}))
        return
    paths = (args.base_a, args.guard_a, args.guard_b, args.base_b, args.out)
    if not all(paths):
        parser.error("all four ABBA inputs and --out are required")
    if not validate_request(".github/fukaji-uct-guard-ab-request.json"):
        raise ValueError("research request is disabled")
    print(json.dumps(audit(*paths), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
