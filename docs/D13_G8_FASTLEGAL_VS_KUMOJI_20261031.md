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

## 重要：初回比較のシード交絡と再実験（2026-10-31）

コード監査で、初回run `41184358423` は同じ序盤・同じ先後でも、雲路に送るUSI `RandomSeed` が候補ON/OFFで異なる実装を発見（`NEW_SEED + pair*16 + arm_index*4 + seat*2`）。雲路のrunner `configure_game()` は確かに `setoption name RandomSeed value ...` を送信するため、同一相手条件という比較の厳密性が欠ける。**初回runの勝敗は単一要因の候補採否に使わない**。雲路・深路の速度版どちらにも必要な同じ乱数値を渡すよう `matched_seed(pair,seat)` を実装（`NEW_SEED + pair*16 + seat*2`）、ゲーム単位にも実seedを記録し集計で検査する。追加の回帰テストでこの不変条件を維持。さらにrun実態の上書き回避と同じ目的の旧run自動取消しのため、Actions concurrency を新run優先に修正。リクエストに厳格 `protocol_revision=identical_randomseed_fix_20261031` を追加して再起動要求commit `a0deb65bf499012c4a6ff0c629c982117462b372` を保存。**修正後の新run IDと200局の成績は、実状態が確認できるまで未確認。** 初回runの一部ジョブをcancelledと確認済み。


## D13修正版200局の確定監査（2026-10-31 JST17:41以降）
有効run [41184685991](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/41184685991)、request SHA `a0deb65bf499012c4a6ff0c629c982117462b372`。Actionsはsuccess。announce/preflight/prepare、20/20 games shard、aggregate、final_statusの全26 job成功。入力1、shard20、結果1の計22 artifactsを確認。`research/d13-kumoji-cross200/run-status.json`はverified_200_complete、summaryは全200局合法再生、両arm同一RandomSeed・同一雲路binary/settings・G8モデルSHA照合成功を記録。旧run41184358423はRandomSeed交絡のため採否から除外。

確定結果：FastLegalGuard OFF 100局＝深路0勝/雲路100勝/未決着0、ON 100局＝深路0勝/雲路100勝/未決着0。50ペアすべてequal、符号検定は有効な不一致ペア0のためp値なし。深さ3完了OFF165/2162手、ON320/2271手、深さ2はOFF1566/ON1772、深さ1はOFF431/ON179。雲路知識ヒットOFF0/4060 probe、ON0/4278 probe。G8モデルSHA256 `18b39f647caa458389a1900b875337adadac74ae37e27343e7f0094ed663c492`、雲路binary SHA256 `2e4ad04ac7fee914573f8a369cf10724701891041cfa32aa59741a8390eab9ff`、知識SHA256 `082f2aaf58201e44f6fa786a6253a2584e9191cbaf55d13f49d56e57198dde2a`、全棋譜SHA256 `ab29882f059166427d369ea033773fee63360a6916244ad661d097a0b7c0ea58`。

判断：D12の独立240局での高速化優位は保持するが、D13では雲路相手の棋力改善は実証されず、正式採用しない。雲路は2スレッド・長考/先読みを使い、深路と対称な計算資源ではない。次は同一G8/同一高速化設定で王手応手・末端取り返しの単独仮説を固定局面診断→別seed40局→有望時240局で検証する。対雲路0勝を手数延長だけで改善認定しない。AddnessはFORBIDDENのため再試行せず未保存。既定USI・正式G8は不変更。
