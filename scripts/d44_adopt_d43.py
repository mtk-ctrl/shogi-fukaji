"""D44: atomically prepare D43 NNUE weights and current model metadata for CI.

This script edits the local checkout only, and never pushes directly.
GitHub Actions validates the changes and performs a single guarded push.
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

ROOT=Path(__file__).resolve().parents[1]
OLD="170d6c1ee41d77b69f0b75722ece3deb85f3eea189f70d9789db477a38266db1"
NEW="bf8c26f9f8f6adff8a87c46ab04fcf5cd76a7915be241da11b99bb4ae2ac8b9f"
TRAIN_ARCHIVE_SHA="955a1785940f9f2aac946739af1b8637e5ef53d06297e6ad7d32b75a5a94a236"
AUDIT_ARCHIVE_SHA="c08ccab3f5e9ef90c288e8ccfc5f3951eb188dfc30876af775c2ba3cfb03f510"
TRAIN_SOURCE_SHA="fe0a4eaec246a7e2e32c911c4327232eef83f8a87897e7a3a96538220ced6bbd"
RELEASE_RUN="42292240321"
WEIGHTS_PATH="models/fukaji-d43.nnue.npz"
DECISION_PATH="docs/decisions/2026-11-07_D43学習後モデルを現行に採用.md"
DATE="2026-11-07"

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def check(path,expected):
    actual=sha(path.read_bytes())
    if actual!=expected:
        raise ValueError(f"mismatched SHA256: {path}, observed {actual}, expected {expected}")

def read_tar(tar,name):
    member=tar.extractfile(name)
    if member is None:raise ValueError(f"missing {name}")
    return member.read()

def patch(path,old,new):
    p=ROOT/path
    source=p.read_text()
    if old not in source:
        raise ValueError(f"expected pre-update text missing in {path}: {old[:80]}")
    p.write_text(source.replace(old,new,1))

def atomic_json(path,data):
    p=ROOT/path;p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n")
    tmp.replace(p)

def prepare(trained,audit,out):
    check(trained,TRAIN_ARCHIVE_SHA)
    check(audit,AUDIT_ARCHIVE_SHA)
    original=json.loads((ROOT/"formal/model.json").read_text())
    if (original["research_snapshot_sha256"]!=OLD or
        original["optimizer_step"]!=2106 or
        original["version"]!="FUKAJI v0.4.0"):
        raise ValueError("baseline changed; do not overwrite another adopted model")
    contract=json.loads((ROOT/"research/d17-fast-self-learning/formal-contract.json").read_text())
    for name,old_hash in contract.items():
        if sha((ROOT/name).read_bytes())!=old_hash:
            raise ValueError("existing formal source/contract already drifted: "+name)
    with tarfile.open(trained,"r:gz") as t:
        training=json.loads(read_tar(t,"report.json"))
        if (training.get("schema")!="d43-learned1000-v1" or
            training.get("baseline")!=OLD or training.get("trained_sha256")!=NEW or
            training.get("optimizer_step_before")!=2106 or
            training.get("optimizer_step_after")!=2882 or
            training.get("new_games")!=1000 or
            training.get("resolved_new_games")!=994 or
            training.get("source_sha256")!=TRAIN_SOURCE_SHA):
            raise ValueError("D43 training model, optimizer or 1000-game source inconsistent")
        model_dir=out/"models"
        model_dir.mkdir(parents=True,exist_ok=True)
        (model_dir/f"{NEW}.npz").write_bytes(read_tar(t,f"{NEW}.npz"))
        (model_dir/f"{NEW}.json").write_bytes(read_tar(t,f"{NEW}.json"))
    model,optimizer,_=load_snapshot(model_dir,NEW)
    if (optimizer["step"]!=2882 or model.inputs!=2548 or
        model.hidden_weights.shape!=(2548,32)):
        raise ValueError("D43 trained NNUE tensors/optimizer are invalid")
    with tarfile.open(audit,"r:gz") as t:
        result=json.loads(read_tar(t,"report.json"))
        plan=json.loads(read_tar(t,"plan.json"))
        if (result.get("schema")!="d43-paired200-result-v1" or
            result.get("before")!=OLD or result.get("after")!=NEW or
            result.get("games")!=200 or result.get("opening_pairs")!=100 or
            result.get("before_wins")!=50 or result.get("after_wins")!=150 or
            result.get("draws")!=0 or result.get("unresolved")!=0 or
            result.get("predeclared_verdict")!="after_better" or
            result.get("formal_changed") is not False or
            plan.get("decision",{}).get("margin_win_count")!=12):
            raise ValueError("D43 verified 200 game evidence does not authorize adoption")
        game_hashes=result.get("game_sha256",{})
        if len(game_hashes)!=200 or len(plan.get("families",[]))!=100:
            raise ValueError("incomplete D43 paired100 families")
        for name,expected in game_hashes.items():
            if sha(read_tar(t,"games/"+name))!=expected:
                raise ValueError("D43 comparison record hash mismatch: "+name)
    dest=ROOT/WEIGHTS_PATH
    dest.parent.mkdir(parents=True,exist_ok=True)
    buf=io.BytesIO()
    np.savez_compressed(buf,
        w=model.hidden_weights,b=model.hidden_bias,
        v=model.output_weights,o=np.array([model.output_bias],dtype=np.float32))
    raw=buf.getvalue()
    if dest.exists() and dest.read_bytes()!=raw:
        raise ValueError("existing D43 inference model has different bytes")
    dest.write_bytes(raw)
    weights_sha=sha(raw)
    with np.load(io.BytesIO(raw),allow_pickle=False) as a:
        if set(a.files)!={"w","b","v","o"}:
            raise ValueError("inference model must not contain Adam")
        for key,expected in (
            ("w",model.hidden_weights),("b",model.hidden_bias),
            ("v",model.output_weights)):
            np.testing.assert_array_equal(a[key],expected)
        if float(a["o"][0])!=float(model.output_bias):
            raise ValueError("output bias was not preserved")
    new_spec=dict(original)
    new_spec.update({
        "approved_on":DATE,
        "approved_by":"owner: 2026-11-07 D43 1000-game self-learning and 150-50 paired200 adoption",
        "research_snapshot_sha256":NEW,
        "weights_file":WEIGHTS_PATH,
        "weights_sha256":weights_sha,
        "research_parent_sha256":OLD,
        "optimizer_step":2882,
        "training_source_sha256":TRAIN_SOURCE_SHA,
        "training_release_sha256":TRAIN_ARCHIVE_SHA,
        "comparison_release_sha256":AUDIT_ARCHIVE_SHA,
        "model_distribution":"GitHub source models/fukaji-d43.nnue.npz plus public D44 runnable release; full trained optimizer retained in D43 learned Release.",
        "validation":{
            "opponent":"D37 frozen before training (170d6c1e); 100 held-out openings x swapped colors",
            "wins":150,"losses":50,"draws":0,"unresolved":0,
            "opening_families":100,"games":200,
            "planned_opening_families":100,"planned_games":200,
            "normal_move_time_ms":200,"longthink_ms":1000,
            "longthink_max_per_side":10,"quiescence":0,
            "strength_conclusion":"D43 after 1000 training games wins 150-50 over D37 in standard 200ms plus longthink; adoption threshold satisfied"
        }
    })
    # The version/USI identifier and search parameters are deliberately untouched:
    # those changes need a separate owner authorization under F00/F20.
    for key in ("version","backend","quiescence","default_move_time_ms",
                "default_search_depth","architecture"):
        if new_spec[key]!=original[key]:
            raise ValueError("formal USI/search contract drift: "+key)
    atomic_json("formal/model.json",new_spec)
    loaded,loaded_spec=load_formal()
    if loaded_spec["weights_sha256"]!=weights_sha:
        raise ValueError("formal_model.load rejected the inference model")
    np.testing.assert_array_equal(loaded.hidden_weights,model.hidden_weights)

    decision=f"""# 2026-11-07：1000局学習後D43 NNUEを現行モデルへ採用

## 採用判断

本人から「新モデルを現行化して下さい」と明示指示。深路F00/F10/F20に従い、**学習前D37から直接更新したD43（NNUE {NEW}、Adam2882）を現行重みとして採用**。別の研究モデル系列は作らない。F00に従い正式USI名・version・探索仕様・既定時刻は変更しない（version表示は FUKAJI v0.4.0 のまま）。

## 事前に凍結した実験と検証

- [D43 run](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/{RELEASE_RUN})：5wave×独立20ジョブ×各10局＝**1000局**、994局確定・6局未決着。完全合法再生とSHA、[1000局Release](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d43-d37-1000-selfplay-{RELEASE_RUN})（SHA256 `{TRAIN_SOURCE_SHA}`）を確認。
- 1000局で学習済みの重みはD37（`{OLD}`、Adam2106）から直接継承し、既存合法train665局をreplay、学習局面49,613、2epoch、Adam2106→2882。学習後の[完全重み・optimizer Release](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d43-d37-trained-{RELEASE_RUN})（アーカイブSHA256 `{TRAIN_ARCHIVE_SHA}`）。
- 100新規独立序盤×先後交替＝200局、学習後D43 **150勝**、学習前D37 **50勝**、引分0・未決着0、25,654手合法再生、1手200ms／1秒長考各陣営最大10回。事前判定閾値「勝数差12以上、未決着10以下」を大幅超過。[全200局・plan・report Release](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d43-d37-before-after-200games-{RELEASE_RUN})（アーカイブSHA256 `{AUDIT_ARCHIVE_SHA}`）を照合。
- 推論用の4テンソルのみを学習後snapshotから抽出し、optimizerを含まない`{WEIGHTS_PATH}`へ保存。重みSHA256 `{weights_sha}`。全テンソルの一致・有限値・標準`formal_model.load`読込を検証。
- 採用はNNUE重みの切替のみ。USIの表記 FUKAJI v0.4.0、探索方式と既定設定を維持する。前のD37完全snapshot／Adam2106はGitHubのD41凍結bundleと既存Drive保全から復元可能。

## 現行化と検証の完了条件

`formal/model.json`、GitHubに追跡された推論ファイル、`docs/現行エンジン仕様.md`、共通ルーター入口、`research/d17-fast-self-learning/formal-contract.json`を同一commitで反映し、全契約テスト、実C++ NNUE・USI対局スモーク、起動パッケージRelease、ハッシュ再取得照合を実施する。途中失敗時に正式mainを更新しない。実現したcommit/run IDと検査receiptはD44実行結果に記録する。

次の自己学習の**唯一の親は現行D43 `{NEW}`、Adam2882**。D41の不採用モデルや旧D37から別研究を再開しない。
"""
    p=ROOT/DECISION_PATH
    if p.exists():
        raise ValueError("D44 decision already exists; refusing to rewrite")
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(decision)

    spec_path="docs/現行エンジン仕様.md"
    p=ROOT/spec_path
    s=p.read_text()
    if "現行モデルは FUKAJI v0.4.0 / D37 NNUE" not in s:
        raise ValueError("current engine specs already modified")
    s=s.replace(
        "更新：2026-11-04。**現行モデルは FUKAJI v0.4.0 / D37 NNUE**。本人がD37を現時点の基準として採用し、次の自己対局強化はD37のAdam2106から直接始める。D37対D36の100ms/200局で勝ち越したが、200ms+長考1秒10回条件の採用時棋力比較は未実施。「正式版」は旧称として扱う。",
        f"更新：{DATE}。**現行のNNUE重みはD43（{NEW[:8]}、Adam2882）**。旧現行D37を1000局の自己対局で追加学習し、正規の200ms＋長考1秒10回の200局直接比較で**150勝50敗**。既定のUSI表記・versionはFUKAJI v0.4.0のまま、探索コードや配布形式は不変。「正式版」は旧称。"
    )
    s=s.replace(f"`{OLD}`",f"`{NEW}`",1)
    s=s.replace("`models/fukaji-v0.4.0.nnue.npz`",f"`{WEIGHTS_PATH}`",1)
    s=s.replace("`2a0001826c2df835b063c896f8f52e2e99972e144921b1ceff6b77ed6e104b8e`",f"`{weights_sha}`",1)
    s=s.replace("重みは本人限定の起動パッケージに同梱する。公開リポジトリ単独には学習済み重みを置かない。",
                "今回採用した推論重みはGitHubの`models/`に保持し、公開起動パッケージにも同梱する。旧D37配布は復元用に保持。",1)
    a=s.index("## 現行モデルの採用判断と強さの不確実性")
    b=s.index("## 復元・研究との境界",a)
    s=s[:a]+f"""## 現行モデルの採用判断と強さの不確実性

[D43採用記録](decisions/2026-11-07_D43学習後モデルを現行に採用.md)により、学習後D43（`{NEW}`、Adam2882）を現行NNUEとする。D37からの新規自己対局1000局、旧train665局のreplay、2epoch追加学習。新規100序盤先後交替200局（200ms＋各陣営1秒長考10回）でD43は**150勝50敗、引分0、未決着0**。実際の棋力向上の根拠は学習lossではなくこの対局結果。ただし他環境・他時間制御の勝率を保証するものではない。原棋譜と比較条件は[GitHub Release](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d43-d37-before-after-200games-{RELEASE_RUN})に保存済み。

""" + s[b:]
    s=s.replace("NNUE重みは現行モデルD37（Adam2106）の完全stateから直接学習し、更新前snapshotと更新後の重みを同条件で対局させる。",
                "次のNNUE自己学習は現行D43（Adam2882）のGitHub保存済み完全stateから開始し、更新前snapshotと更新後を同条件で対局比較する。")
    p.write_text(s)

    a="**現行モデルはv0.4.0/D37 NNUE（170d6c1e、Adam2106）**"
    b=f"**現行NNUE重みはD43（{NEW[:8]}、Adam2882）。USI表示はFUKAJI v0.4.0を維持**"
    patch("AGENTS.md",a,b)
    router=ROOT/"docs/00_開発ルーター.md"
    s=router.read_text()
    a=s.index("## 現行モデル")
    b=s.index("## 現在の研究・保存再開点",a)
    s=s[:a]+f"""## 現行モデル
**現行NNUEはD43（{NEW[:8]}、Adam2882）、旧D37から1000局自己学習後の採用重み**。USI表示 FUKAJI v0.4.0と探索仕様は変更しない。次周は[D43現行化の決定](decisions/2026-11-07_D43学習後モデルを現行に採用.md)と[現行仕様](現行エンジン仕様.md)、[公開manifest](../formal/model.json)を参照して、必ずこの現行重みとoptimizerから学習開始。学習前の重みは不変snapshotで保持し、学習後と直接比較して次の採否を決定。旧D37および他研究候補を学習親に使わない。公開GitHubにD43推論重みを保存し、完全重み・optimizerは[学習Release](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d43-d37-trained-{RELEASE_RUN})に保存。

""" + s[b:]
    router.write_text(s)
    roadmap=ROOT/"docs/ROADMAP.md"
    s=roadmap.read_text()
    s=s.replace("2026-11-04更新。現行状態は以下。",f"{DATE}更新。現行状態は以下。",1)
    a=s.index("## 現行モデル")
    b=s.index("## 現在の研究",a)
    s=s[:a]+f"""## 現行モデル

**D43（NNUE `{NEW[:12]}…`、Adam2882）を現行採用。USIのversion表示はFUKAJI v0.4.0を維持。** 旧D37（Adam2106）を直接追加学習：新規1000自己対局・決着994局、合法train＋過去665局、学習49,613局面。学習前D37との**200局標準比較で150勝50敗・引分0・未決着0**。[採用判断](decisions/2026-11-07_D43学習後モデルを現行に採用.md)、[200局の全棋譜Release](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d43-d37-before-after-200games-{RELEASE_RUN})、[D43学習stateとAdam2882](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d43-d37-trained-{RELEASE_RUN})。正式[manifest](../formal/model.json)と[仕様](現行エンジン仕様.md)を正本とする。次回の自己学習は必ずD43が唯一の親。旧D37・e0などは履歴・復元用として保護。

""" + s[b:]
    s+=f"""
## {DATE}：D44、D43学習後NNUEの現行化

- 本人指示によりD43(`{NEW[:8]}`/Adam2882)を現行モデルとして反映。[決定と採否根拠](decisions/2026-11-07_D43学習後モデルを現行に採用.md)。
- 推論用`{WEIGHTS_PATH}`のSHA256：`{weights_sha}`。D43完全学習stateとoptimizerはGitHub Releaseに保存済み、旧D37は復元用に残す。
- 現行のUSI/version・探索コード・既定の探索時間は従来どおり。重みとoptimizerの進行状態だけを更新。正式契約テスト、C++ NNUE/USI動作と配布・SHAの完了証拠はD44 GitHub Actions runのログ／現行Releaseを参照。
"""
    roadmap.write_text(s)

    # The integrity lock covers exactly these two intentionally changed files.
    for name in ("formal/model.json","docs/現行エンジン仕様.md"):
        contract[name]=sha((ROOT/name).read_bytes())
    atomic_json("research/d17-fast-self-learning/formal-contract.json",contract)
    out.mkdir(parents=True,exist_ok=True)
    receipt={"schema":"fukaji-d44-adopted-prepare-v1","parent":OLD,"current":NEW,
             "version_unchanged":new_spec["version"],
             "inference_file":WEIGHTS_PATH,"inference_sha256":weights_sha,
             "optimizer_step":2882,"train_sha256":TRAIN_ARCHIVE_SHA,
             "arena_sha256":AUDIT_ARCHIVE_SHA,"arena_wins":150,
             "arena_losses":50,"full_state_preserved":True,
             "contract_sha256":sha((ROOT/"research/d17-fast-self-learning/formal-contract.json").read_bytes())}
    (out/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt))
    
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--trained",type=Path,required=True)
    parser.add_argument("--audit",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    prepare(args.trained,args.audit,args.out)

if __name__=="__main__":
    main()
