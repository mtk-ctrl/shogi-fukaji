# D13：雲路と同じ条件で深路G8の高速化ON/OFFを200局比較（2026-10-31）

## 狙い
過去に雲路の全研究候補と対局して深路G8が100局すべて負けたことを受け、深路自身のFastLegalGuard改善が雲路相手に意味を持つかを、従来G8と新候補で同じ条件で比較する。旧100局の序盤を流用せず**新seed50序盤を各4局（G8 OFF/ON×先手/後手）**。200局対局。深路と雲路の条件は非対称だが、ON/OFF間で同じ雲路設定を固定する。

## 凍結した内容
- 実行：GitHub Actions run `41184358423`（2026-10-31 JST17:22頃に起動）。commit `844f86d391e89a4c1739c769b8ac1afd757c77eb`。
- 元の雲路研究候補は以前の `run41177748192` の入力artifact ID `13033349670` を再利用。雲路ソース commit `76ee74d04a800285e24dd81ef01711e990cfa181`、研究候補知識460局面、G8モデルSHA `18b39f647caa458389a1900b875337adadac74ae37e27343e7f0094ed663c492`。Prepareで入力receiptのSHAとモデル・バイナリ・局面知識ファイルを照合。
- 雲路：PositionKnowledge ON（全研究由来の460局面、正式採用データではない）、2スレッド・AdaptiveLongThink ON・Ponder ON・RiskVerify ON、`go movetime 200`。深路：G8相対mix25/上限3/高速相対評価/root alpha beta/FastMoveList ON、`go movetime 200`、唯一の差分はFastLegalGuard OFF（基準）/ON（候補）。
- 新seed `202611015`、旧seed `202610315` を序盤と照合して重複を拒否。全200局を20分割×最大10並列で実対局。毎局全着手・各手の探索統計・終局判定を保存し、全20shardのSHAと200ゲームID・先後・開幕を監査して合計結果を出す。
- プロトコル・研究専用本体：`fukaji/kumoji_cross200.py`、入口 `.github/workflows/kumoji-cross200.yml`、要求 `.github/fukaji-kumoji-cross200-request.json`、軽量回帰 `tests/test_kumoji_cross200.py`。

## 2026-10-31 JST17:25の進捗
announce、preflight、prepareの実GitHub Actions jobはすべて成功。200局のゲームjobは20分割のうち10本が進行中、10本がqueue。正式採用なし。勝敗・最終結果は未確認。結果正本予定 `research/d13-kumoji-cross200/summary.json` と実行状態 `run-status.json`、完了棋譜は期限付きActions artifact。

## 終了後に判定する項目
1. 旧知識460局面と新モデルSHA、雲路binary/プロトコルが従来と同じこと。
2. 新規50組すべて先後同じ初期手順、標準/高速が各100局を完了したこと。全手合法再生と真正な詰み勝敗・未決着を区別する。
3. 雲路相手の勝利数だけでなく、50組のfastlegal ON/OFF直接優劣、深路の探索深度、雲路の知識ヒット、各局の最終手数・終局理由。両者0勝でも負けた手数の単純増減を棋力改善とは認定しない。
4. 研究データを雲路の教師手として転用しない。正式版採用の判断は本人に報告し、無断採用しない。
5. 強化を止めず、雲路への改善が不十分なら深路の王手への応手・3手以降の取り合いを別研究として進める。時間超過の問題は別の単一要因研究に分ける。

## 記録の接続制限（2026-10-31）
GitHubの研究記録は保存・読戻し済み。深路Addness Goal `ab88e21e-083f-41b3-8b65-daf39872a2e1` への新しいD12確定結果・D13進行状況の差分保存は、正規Addnessコネクタで `FORBIDDEN: Blocked by a security check before reaching Addness. Nothing was saved.` の拒否となり未保存。権限制約を別経路や書換えで回避せず、実験・GitHub上の正本は独立に維持する。
