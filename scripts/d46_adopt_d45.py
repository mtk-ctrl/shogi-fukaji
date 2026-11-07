"""D46: adopt the D45 second-round learned NNUE only after frozen evidence checks.

This prepares a local checkout; the CI workflow validates everything and then
performs one fast-forward commit. It never pushes from Python.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from fukaji.online_learning import load_snapshot
from fukaji.formal_model import load as load_formal

ROOT = Path(__file__).resolve().parents[1]
OLD = "bf8c26f9f8f6adff8a87c46ab04fcf5cd76a7915be241da11b99bb4ae2ac8b9f"
NEW = "941598f2a9ec926d5f67dadf03582ccef718a56a843070dc1f576cc7eeca44e8"
SOURCE_SHA = "be22c4e92eff0b684f06502dcdf9be52071ab60beab5a4a91edd8605f4d873c4"
TRAIN_SHA = "d44349470b4068224ce1383f29d1afdf941c39931f6141c278927976da52c820"
AUDIT_SHA = "6531747f10d015bec7817186f6fbed149aa6ac1822d94fbcce7858a7ceae211b"
RUN = 42326260509
WEIGHTS_PATH = "models/fukaji-d45.nnue.npz"
DECISION_PATH = "docs/decisions/2026-11-07_D45学習後モデルを現行に採用.md"


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def checked_archive(path: Path, expected: str) -> tarfile.TarFile:
    actual = digest(path.read_bytes())
    if actual != expected:
        raise ValueError(f"archive SHA256 mismatch: {path}: {actual}")
    return tarfile.open(path, "r:gz")


def read_member(t: tarfile.TarFile, name: str) -> bytes:
    file = t.extractfile(name)
    if file is None:
        raise ValueError("missing archive member " + name)
    return file.read()


def replace_exact(path: str, old: str, new: str) -> None:
    p = ROOT / path
    s = p.read_text()
    if old not in s or s.count(old) != 1:
        raise ValueError(f"expected unique current text absent: {path} {old[:80]}")
    p.write_text(s.replace(old, new, 1))


def replace_section(path: str, first: str, last: str, replacement: str) -> None:
    p = ROOT / path
    s = p.read_text()
    a = s.index(first)
    b = s.index(last, a)
    p.write_text(s[:a] + replacement + s[b:])


def write_json(path: str, obj: dict) -> None:
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def prepare(trained: Path, audit: Path, out: Path) -> None:
    original = json.loads((ROOT / "formal/model.json").read_text())
    if (original["research_snapshot_sha256"] != OLD or
            original["optimizer_step"] != 2882 or
            original["weights_file"] != "models/fukaji-d43.nnue.npz" or
            original["version"] != "FUKAJI v0.4.0"):
        raise ValueError("baseline is not current D43; refuse concurrent overwrite")
    contract_path = "research/d17-fast-self-learning/formal-contract.json"
    contract = json.loads((ROOT / contract_path).read_text())
    for name, expected in contract.items():
        if digest((ROOT / name).read_bytes()) != expected:
            raise ValueError("formal source/contract drift: " + name)

    out.mkdir(parents=True, exist_ok=True)
    model_dir = out / "models"
    model_dir.mkdir(exist_ok=True)
    with checked_archive(trained, TRAIN_SHA) as t:
        report = json.loads(read_member(t, "report.json"))
        requirements = {
            "schema": "d45-learned1000-v1", "baseline": OLD,
            "trained_sha256": NEW, "optimizer_step_before": 2882,
            "optimizer_step_after": 3794, "new_games": 1000,
            "resolved_new_games": 994, "old_bootstrap_replay_games": 665,
            "prior_d43_replay_games": 300, "source_sha256": SOURCE_SHA,
            "strength_proven": False, "formal_model_changed": False
        }
        for name, value in requirements.items():
            if report.get(name) != value:
                raise ValueError(f"trained D45 {name} mismatch: {report.get(name)!r}")
        for ext in ("npz", "json"):
            (model_dir / f"{NEW}.{ext}").write_bytes(read_member(t, f"{NEW}.{ext}"))
    model, optimizer, _ = load_snapshot(model_dir, NEW)
    if (optimizer is None or optimizer["step"] != 3794 or model.inputs != 2548 or
            model.hidden_weights.shape != (2548, 32)):
        raise ValueError("D45 learned NNUE structure/Adam not verified")

    with checked_archive(audit, AUDIT_SHA) as t:
        result = json.loads(read_member(t, "report.json"))
        plan = json.loads(read_member(t, "plan.json"))
        requirements = {
            "schema": "d45-paired200-result-v1", "before": OLD,
            "after": NEW, "games": 200, "opening_pairs": 100,
            "before_wins": 44, "after_wins": 155, "draws": 0,
            "unresolved": 1, "difference": 111,
            "predeclared_verdict": "after_better", "formal_changed": False
        }
        for name, value in requirements.items():
            if result.get(name) != value:
                raise ValueError(f"D45 audit {name} mismatch: {result.get(name)!r}")
        if (plan.get("schema") != "d45-paired200-v1" or
                plan.get("before") != OLD or plan.get("after") != NEW or
                plan.get("games") != 200 or plan.get("pairs") != 100 or
                plan.get("decision", {}).get("margin_win_count") != 12 or
                plan.get("decision", {}).get("max_unresolved") != 10 or
                len(plan.get("families", [])) != 100):
            raise ValueError("D45 frozen audit plan mismatch")
        conditions = plan.get("conditions", {})
        if any(conditions.get(key) != value for key, value in
                (("backend", "cpp_nnue"), ("quiescence", 0),
                 ("normal_ms", 200), ("long_ms", 1000),
                 ("max_long_per_side", 10))):
            raise ValueError("D45 audit clock / engine differs")
        hashes = result.get("game_sha256", {})
        if len(hashes) != 200 or result.get("validated_plies") != 24451:
            raise ValueError("missing game records or legality audit")
        for name, expected in hashes.items():
            if digest(read_member(t, "games/" + name)) != expected:
                raise ValueError("D45 full-audit individual game hash mismatch: " + name)

    dest = ROOT / WEIGHTS_PATH
    if dest.exists():
        raise ValueError("D45 official weights already exist; refuse overwrite")
    memory = io.BytesIO()
    np.savez_compressed(memory, w=model.hidden_weights, b=model.hidden_bias,
                        v=model.output_weights,
                        o=np.array([model.output_bias], dtype=np.float32))
    raw = memory.getvalue()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(raw)
    weights_sha = digest(raw)
    with np.load(io.BytesIO(raw), allow_pickle=False) as values:
        if set(values.files) != {"w", "b", "v", "o"}:
            raise ValueError("inference weights contain unexpected tensors")
        for key, arr in (("w", model.hidden_weights), ("b", model.hidden_bias),
                         ("v", model.output_weights)):
            np.testing.assert_array_equal(values[key], arr)
            if not np.all(np.isfinite(values[key])):
                raise ValueError("nonfinite NNUE tensor")
        if float(values["o"][0]) != float(model.output_bias):
            raise ValueError("NNUE output bias mismatch")

    spec = dict(original)
    spec.update({
        "approved_on": "2026-11-07",
        "approved_by": "owner: 2026-11-07 D45 second-round learned NNUE adoption, 155-44 paired200 and 1 unresolved",
        "research_snapshot_sha256": NEW,
        "research_parent_sha256": OLD,
        "weights_file": WEIGHTS_PATH,
        "weights_sha256": weights_sha,
        "optimizer_step": 3794,
        "training_source_sha256": SOURCE_SHA,
        "training_release_sha256": TRAIN_SHA,
        "comparison_release_sha256": AUDIT_SHA,
        "model_distribution": "GitHub source models/fukaji-d45.nnue.npz plus public D46 runnable release; full optimizer in D45 trained Release.",
        "validation": {
            "opponent": "D43 frozen before D45 training (bf8c26f9); 100 held-out openings x swapped colors",
            "wins": 155, "losses": 44, "draws": 0, "unresolved": 1,
            "opening_families": 100, "games": 200,
            "planned_opening_families": 100, "planned_games": 200,
            "normal_move_time_ms": 200, "longthink_ms": 1000,
            "longthink_max_per_side": 10, "quiescence": 0,
            "strength_conclusion": "D45 after second 1000 training games wins 155-44 over D43 (one unresolved) in paired 200; preset margin satisfied"
        }
    })
    for key in ("version", "backend", "quiescence", "default_move_time_ms",
                "default_search_depth", "architecture"):
        if spec[key] != original[key]:
            raise ValueError("not authorized to change USI/search: " + key)
    write_json("formal/model.json", spec)
    loaded, check_spec = load_formal()
    if check_spec["weights_sha256"] != weights_sha:
        raise ValueError("formal NNUE loader rejected newly pinned model")
    np.testing.assert_array_equal(loaded.hidden_weights, model.hidden_weights)

    decision = f"""# 2026-11-07：D45第2周の強化重みを現行モデルに採用

本人の明示指示「さっきの強化モデルを現行モデルとして反映」により、D43から直接自己対局・追加学習したD45重み \`{NEW}\`（Adam3794）を現行とする。独立した研究系列は作らない。USI表示 FUKAJI v0.4.0、探索コード・設定・配布形式は変えない。

- 更新前：D43 \`{OLD}\`、Adam2882。旧推論ファイルおよび完全学習stateを保全。
- 自己対局：新1000局（994決着・6未決着）、学習58,334局面・2epoch、Adam2882→3794。[D45実行](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/{RUN})、[学習完全state](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d45-d43-trained-{RUN}) SHA256 \`{TRAIN_SHA}\`。
- 学習前後直接比較：独立100序盤×先後交替200局、1手200msと1秒長考最大10回/陣営。**D45 155勝、D43 44勝、引分0、未決着1**。24,451手合法再生、事前基準（勝数差12以上かつ未決着10以下）通過。未決着を勝敗へ算入しない。[比較全棋譜](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d45-d43-before-after-200games-{RUN}) SHA256 \`{AUDIT_SHA}\`。
- 研究snapshot \`{NEW}\`、推論 \`{WEIGHTS_PATH}\` SHA256 \`{weights_sha}\`。完全重み・optimizerは学習Releaseにあり、次周の唯一の親。
- GitHub ActionsのD46採用ジョブで全契約・回帰テスト・実C++/USIと配布読戻し・SHAまで検証する。成功したコミットとrunはActions及び現行Releaseを採用完了の証拠とする。失敗時はD43を維持する。

人間の定跡を教師として注入せず、詰みはルールから正確に読む方針を継続する。
"""
    path = ROOT / DECISION_PATH
    if path.exists():
        raise ValueError("D45 decision already exists")
    path.write_text(decision)

    specpath = "docs/現行エンジン仕様.md"
    replace_exact(specpath,
        "更新：2026-11-07。**現行のNNUE重みはD43（bf8c26f9、Adam2882）**。旧現行D37を1000局の自己対局で追加学習し、正規の200ms＋長考1秒10回の200局直接比較で**150勝50敗**。",
        "更新：2026-11-07。**現行のNNUE重みはD45（941598f2、Adam3794）**。D43から第2周の1000局自己対局で追加学習し、200ms＋長考1秒10回の200局直接比較で**155勝44敗・未決着1局**。")
    replace_exact(specpath, f"\`{OLD}\`", f"\`{NEW}\`")
    replace_exact(specpath, "\`models/fukaji-d43.nnue.npz\`", f"\`{WEIGHTS_PATH}\`")
    replace_exact(specpath,
        "\`f6ba1d5af46ae648b1921d675ada28cf5065f23a20e424154458817534bf42a6\`",
        f"\`{weights_sha}\`")
    replace_section(specpath,
        "## 現行モデルの採用判断と強さの不確実性",
        "## 復元・研究との境界",
        f"""## 現行モデルの採用判断と強さの不確実性

[D45採用記録](decisions/2026-11-07_D45学習後モデルを現行に採用.md)。D43/Adam2882を直接追加学習したD45/Adam3794を現行とする。新規自己対局1000局（994確定、6未決着）、旧train再利用を含む58,334局面・2epoch。事前固定の100序盤先後交替200局で**D45が155勝44敗・引分0・未決着1**。比較は通常200msと最大10回の1秒長考、静止探索0。未決着は勝敗扱いしない。他の時計・相手で同じ勝率を保証するものではない。[完全棋譜・評価Release](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d45-d43-before-after-200games-{RUN})。

""")
    replace_exact(specpath,
        "次のNNUE自己学習は現行D43（Adam2882）のGitHub保存済み完全stateから開始し、更新前snapshotと更新後を同条件で対局比較する。",
        "次のNNUE自己学習は現行D45（Adam3794）のGitHub保存済み完全stateから開始し、更新前snapshotと更新後を同条件で対局比較する。")
    replace_exact(specpath,
        "失敗・悪化時は更新前D37へ戻す。",
        "失敗・悪化時は更新前D45へ戻す。")
    replace_exact("AGENTS.md",
        "**現行NNUE重みはD43（bf8c26f9、Adam2882）。USI表示はFUKAJI v0.4.0を維持**",
        "**現行NNUE重みはD45（941598f2、Adam3794）。USI表示はFUKAJI v0.4.0を維持**")
    replace_section("docs/00_開発ルーター.md",
        "## 現行モデル", "## 現在の研究・保存再開点",
        f"""## 現行モデル
**現行NNUEはD45（\`{NEW}\`、Adam3794）**。D43を直接1000局追加学習し、200局標準比較で155勝44敗、未決着1で採用。USI v0.4.0・探索仕様は不変。[採用決定](decisions/2026-11-07_D45学習後モデルを現行に採用.md)、[現行仕様](現行エンジン仕様.md)、[manifest](../formal/model.json)を正本とする。次周は[D45の完全重み・Adam3794](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d45-d43-trained-{RUN})を唯一の学習親とし、旧D43は不変の比較・復元用snapshotとして保全する。独立候補系列を作らない。

""")
    replace_section("docs/ROADMAP.md", "## 現行モデル", "## 現在の研究",
        f"""## 現行モデル

**D45 NNUE（\`{NEW}\`、Adam3794）を現行採用。USI表示 FUKAJI v0.4.0は不変。** D43（\`{OLD}\`、Adam2882）から自己対局1000局と58,334局面追加学習し、独立100序盤×先後200局で**155勝44敗・引分0・未決着1**。[採用決定](decisions/2026-11-07_D45学習後モデルを現行に採用.md)、[比較結果](D45_D43_SECOND_ROUND_1000_GAMES_RESULT_20261107.md)、[完全state](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d45-d43-trained-{RUN})、[manifest](../formal/model.json)・[現行仕様](現行エンジン仕様.md)。次周は必ずD45/Adam3794から。旧D43/D37は復元専用として残す。

""")
    roadmap = ROOT / "docs/ROADMAP.md"
    with roadmap.open("a") as f:
        f.write(f"""
## 2026-11-07：D46によるD45現行化

- 本人指定D45 \`{NEW}\`/Adam3794。200局155勝44敗・未決着1。強化後推論 \`{WEIGHTS_PATH}\` SHA256 \`{weights_sha}\`。
- 変更はNNUE推論重みと現行モデル参照のみ。FUKAJI v0.4.0・探索不変。GitHub Actions D46の全契約・実USI・配布/読戻し完了とmain反映が採用完了条件。
- 完全重み・optimizerはD45学習Release、旧D43推論と完全stateは保持。次周唯一の親はD45/Adam3794。
""")
    for name in ("formal/model.json", specpath):
        contract[name] = digest((ROOT / name).read_bytes())
    write_json(contract_path, contract)
    receipt = {"schema": "fukaji-d46-adopted-prepare-v1",
               "parent": OLD, "current": NEW, "optimizer_step": 3794,
               "inference_file": WEIGHTS_PATH, "inference_sha256": weights_sha,
               "train_archive_sha256": TRAIN_SHA,
               "audit_archive_sha256": AUDIT_SHA, "wins": 155, "losses": 44,
               "draws": 0, "unresolved": 1,
               "version_unchanged": spec["version"],
               "contract_sha256": digest((ROOT / contract_path).read_bytes())}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    options = parser.parse_args()
    prepare(options.trained, options.audit, options.out)
