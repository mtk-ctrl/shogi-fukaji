"""D49: adopt exactly D48 tested search code, with pinned D45 NNUE unchanged.

Only the owner-authorized GitHub Actions workflow calls this one-shot command.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import d48_pvs_mate_compare as d48

SNAPSHOT="941598f2a9ec926d5f67dadf03582ccef718a56a843070dc1f576cc7eeca44e8"
WEIGHTS="9faf66a9d1fee9d282e35835628377ab7951bb94335a5d0498384bf88f4d2c26"
OLD_CPP="06e0bf8c7e9508a400b58955312f277289830c94"
D48_SCRIPT="354b038c7d9247f691e11c46c996e66157ff5977"
D48_MATE="2ad3d9fdeb6126434f5e2c72cd5e9647742382ca"
ARCHIVE_SHA="427ebf9a8671d56b3f3b0fef24a62a2242e293a559e4307027747d30eb3a2f65"
RUN=42339652079

def checksum(p):
    return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()

def gitblob(p):
    return subprocess.check_output(["git","hash-object",p],cwd=ROOT,text=True).strip()

def append(p,block):
    f=ROOT/p
    f.write_text(f.read_text().rstrip()+"\n\n"+block.strip()+"\n")

def main():
    request=json.loads((ROOT/"research/d49-adopt/START.json").read_text())
    expected=dict(
        enabled=True,request_id="d49-20261107-owner-adopt-d48-pvs-mate",
        research_run=RUN,source=OLD_CPP,research_script=D48_SCRIPT,
        mate_source=D48_MATE,research_release_sha256=ARCHIVE_SHA,
        model_snapshot_sha256=SNAPSHOT,weights_sha256=WEIGHTS,
        owner_authorized="formally adopt the completed D48 PVS plus mate search",
        new_usi_version="FUKAJI v0.4.1")
    if request!=expected:raise ValueError("request is not the authorized D49 adoption")
    spec_path=ROOT/"formal/model.json"
    spec=json.loads(spec_path.read_text())
    if (spec["version"]!="FUKAJI v0.4.0" or
        spec["weights_sha256"]!=WEIGHTS or
        spec["research_snapshot_sha256"]!=SNAPSHOT or
        spec["optimizer_step"]!=3794):
        raise ValueError("D45 formal model changed: refusing adoption")
    if gitblob("native/nnue_search_ffi.cpp")!=OLD_CPP:
        raise ValueError("official source no longer D48 original baseline")
    if gitblob("scripts/d48_pvs_mate_compare.py")!=D48_SCRIPT:
        raise ValueError("D48 comparison code changed")
    if gitblob("native/d48_mate_research.inc")!=D48_MATE:
        raise ValueError("D48 exact mate code changed")
    contract_file=ROOT/"research/d17-fast-self-learning/formal-contract.json"
    contract=json.loads(contract_file.read_text())
    for path,expected_hash in contract.items():
        if checksum(path)!=expected_hash:
            raise ValueError("original formal contract differs: "+path)
    if checksum(spec["weights_file"])!=WEIGHTS:raise ValueError("D45 NNUE weights changed")
    # This function generates the byte-exact C++ research candidate used in
    # 100 legal paired games. It refuses to run against a new source revision.
    promoted_cpp=d48.candidate_source()
    (ROOT/"native/nnue_search_ffi.cpp").write_text(promoted_cpp)
    cpp_sha=checksum("native/nnue_search_ffi.cpp")
    spec.update({
       "version":"FUKAJI v0.4.1","approved_on":"2026-11-07",
       "search_backend_variant":"d48_pvs_mate",
       "search_implementation_sha256":cpp_sha,
       "search_adoption":{
          "owner_approval":"2026-11-07: formally adopt D48 PVS plus mate search",
          "research_run":RUN,
          "research_release_archive_sha256":ARCHIVE_SHA,
          "candidate_wins":55,"baseline_wins":44,"draws":0,
          "unresolved":1,"opening_pairs":50,"games":100,
          "research_proven_mates":113,
          "search_features":["PVS","legal forced 1/3/5 ply mate","opponent mate in one root safety"],
          "nnue_weights_unchanged":True}})
    spec_path.write_text(json.dumps(spec,ensure_ascii=False,indent=2)+"\n")
    test=ROOT/"tests/test_native_nnue_search.py"
    old=test.read_text()
    before="                self.assertEqual(ref['nodes'],got['nodes'])"
    if old.count(before)!=1:
        raise ValueError("unexpected PVS node-count legacy assertion")
    test.write_text(old.replace(before,"                self.assertGreater(got['nodes'],0)  # PVS intentionally changes nodes"))
    # D28 accumulator backend intentionally retains old alpha-beta.
    # PVS changes the number of visited nodes but must retain the *exact*
    # same fixed-depth best move and minimax score. Preserve accumulator
    # integrity checks, board restoration, and value comparison.
    acc=ROOT/"tests/test_native_accumulator.py"
    src=acc.read_text()
    old_assert="""                self.assertEqual((a['move'],a['nodes'],a['evaluations']),
                                 (c['move'],c['nodes'],c['evaluations']))"""
    new_assert="""                self.assertEqual(a['move'],c['move'])
                self.assertGreater(a['nodes'],0)
                self.assertGreater(c['nodes'],0)
                self.assertGreater(a['evaluations'],0)
                self.assertGreater(c['evaluations'],0)"""
    if src.count(old_assert)!=1:
        raise ValueError("unexpected accumulator PVS comparison assertion")
    acc.write_text(src.replace(old_assert,new_assert))
    specdoc=ROOT/"docs/現行エンジン仕様.md"
    old=specdoc.read_text()
    a="既定のUSI表記・versionはFUKAJI v0.4.0のまま、探索コードや配布形式は不変。"
    b="D49でPVS＋詰み探索を正式採用し、USI versionはFUKAJI v0.4.1。NNUE重み不変。"
    if old.count(a)!=1:raise ValueError("formal specification header changed")
    old=old.replace(a,b)
    old=old.replace("D23/従来と同じC++基本αβ、静止探索0",
                    "PVS＋1/3/5手の厳密詰み探索・相手1手詰みガード、静止探索0")
    specdoc.write_text(old)
    append("docs/現行エンジン仕様.md",
        """## D49：正式探索へのD48採用（本人承認）

- 現行のUSI名は FUKAJI v0.4.1。C++探索はD48実験と同一のPVS＋合法手・受方全応手による1/3/5手詰み証明＋相手1手詰みを証明した危険候補の回避。
- D48 100局では研究版55勝／現行44勝／未決着1。本人が2026-11-07に正式採用を明示。
- 正式NNUE D45、研究snapshot、推論用重み、Adam3794は不変。go mate 専用USIは従来どおり notimplemented。通常goの詰み補助が今回の採用対象。go depth の固定深さでは詰み補助を実行せずPVSを使用。
- 研究コードとC++内容が完全一致することをD49の準備receiptに記録。従来のD45正式配布Releaseは復元用に不変保存。独立した実USIテスト、契約、隔離展開とSHA再取得をD49 Actionsで検査する。""")
    append("docs/00_開発ルーター.md",
        """## 最新の正式探索（D49、2026-11-07）

USI FUKAJI v0.4.1。NNUEは引き続きD45（941598f2、Adam3794）。D48で100局比較したPVS＋合法詰み1/3/5手＋相手1手詰みガードを本人承認で正式化。上のv0.4.0探索不変という記述はD46時点の歴史的内容。判断記録：docs/decisions/2026-11-07_D49_D48探索を正式採用.md。""")
    append("AGENTS.md",
        """## 最新の正式探索（2026-11-07 D49）

本人はD48のPVS＋詰み探索を正式採用と明示。現行は FUKAJI v0.4.1、D48と同じPVS＋1/3/5手詰み証明＋相手1手詰みガード。NNUEはD45/Adam3794を維持。上部のv0.4.0はD49以前の履歴。判断記録：docs/decisions/2026-11-07_D49_D48探索を正式採用.md。""")
    append("docs/ROADMAP.md",
        """## 2026-11-07 D49：D48の探索を正式採用

本人明示指示で100局実測55勝44敗・未決着1のD48 PVS＋詰み補助を正式USIへ採用。現行USI FUKAJI v0.4.1、NNUE D45 snapshot 941598f2/Adam3794を不変維持。実装・契約検査・隔離配布・SHA検証を実施して正式化する。docs/decisions/2026-11-07_D49_D48探索を正式採用.md。""")
    append("docs/D48_MATE_PVS_PAIRED100_20261107.md",
        """## 後続D49正式採用

本人がD48比較のあと「正式に採用してください。」と明示。D48研究記録は当時の未採用状態のまま保存。D49により、NNUE D45を変えずにD48のC++探索を正式USIへ採用する。判断記録：decisions/2026-11-07_D49_D48探索を正式採用.md。""")
    decision=ROOT/"docs/decisions/2026-11-07_D49_D48探索を正式採用.md"
    if decision.exists():raise ValueError("D49 decision file exists")
    decision.write_text("""# D49：D48のPVS＋詰み探索を正式採用（2026-11-07）

## 本人判断
D48の100局結果報告後、本人が「正式に採用してください。」と明示。探索仕様変更に必要なF00の本人承認を満たす。

## 採用対象
現行D45 NNUE重みは不変。同一重みで実戦比較したD48のPVSと合法手に基づく1/3/5手強制詰みの全応手証明、および相手の1手詰みを証明できる根候補の回避を採用。人間定跡、雲路の評価配点、他AI推奨手は取り込まない。従来の旧探索ソースSHA Git blob 06e0bf8c7e9508a400b58955312f277289830c94を復元可能なGit履歴に維持。USI版をv0.4.1へ更新。

## 根拠と限定
D48 Actions run 42339652079で50独立序盤の先後2局100局：D48研究版55勝、旧正式探索44勝、引分0、未決着1。20ランナーの最大実同時20、全12,395手合法性検証、詰み証明113回。結果は有望だが未知局面での安定優位を保証しない。研究Releaseタグ：fukaji-d48-pvs-mate-d45-100games-42339652079。

## 検証と保存
D49 CIで旧ソースの検証、D48候補のバイト一致、713件を含む回帰、実C++とUSI、隔離配布物のハッシュと起動、GitHub ReleaseのSHA再取得照合を要求。D45完全state/optimizer、旧D45配布物を保持。正式採用の完了報告はD49 CIを確認してから行う。
""")
    append("docs/decisions/2026-11-07_D49_D48探索を正式採用.md",
        """## 初回D49失敗の原因と再発防止
- 初回Actions run 42341568300の713件回帰で12件失敗。すべてtests/test_native_accumulator.pyの古い「旧アルファベータと新PVSの節点数・評価回数も一致」アサーションが原因。実際の最善手は一致し、深さ固定の点数比較は別途維持している。
- PVSは同じ値を少ない節点で計算するため、旧方式と訪問節点数が一致することは**探索正当性の条件ではない**。同試験は最善手と評価点の一致・双方正の節点/評価回数・モデル内部/盤面復元を引き続き検査する。回帰全件再実行と実USI・詰み証明が通るまで正式コードをコミットしない。
- 初回は安全ゲートで停止し、旧D45現行を変更せず。修正後の別runが完了するまで採用完了としない。""")
    # Atomically register owner-approved new SHA on the same staging worktree.
    for p in contract:
        if p!="benchmarks/baseline_ref.txt":contract[p]=checksum(p)
    contract_file.write_text(json.dumps(contract,ensure_ascii=False,indent=2)+"\n")
    if checksum(spec["weights_file"])!=WEIGHTS:raise AssertionError("weights mutated")
    receipt={
        "source_research_run":RUN,
        "new_native_cpp_sha256":cpp_sha,
        "fixed_nnue_sha256":WEIGHTS,
        "fixed_model_snapshot":SNAPSHOT,
        "formal_version":spec["version"],
        "d48_candidate_source_byte_identical":True,
        "previous_native_git_blob":OLD_CPP,
    }
    out=ROOT/"build/d49-adoption-prepare.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(receipt,ensure_ascii=False))

if __name__=="__main__":main()
