# UCT探索木解放費用予約の非計測ABBA実測結果（2026-10-29）

状態：**研究結果のみ・正式採用なし**。雲路main、深路正式USI、G8モデル、既定 `cleanup_aware_reserve=False` は変更していない。

## 実験と監査の実体

- [入口PR#78](https://github.com/mtk-ctrl/shogi-fukaji/pull/78) branch CI [40753142049](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/40753142049) 369/369成功→squash研究main `beaa86252035eb57ba7e4678214b76415a7f8f1c`→main CI [40753454236](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/40753454236) 369/369成功。既定OFF [40753454597](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/40753454597) は重い処理skip。
- 所有27repoの直近最大100runを調査。100件超の8repoはqueued/in_progress/pending/waiting/requestedの5状態を追加照合。重いrun前active 0、同目的run/Issue重複なし。他run停止なし。
- 別明示request commit `adfcb56032053d5f8deeb0e1e49146c9382dccc7` から1runner [run40753877836](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/40753877836) completed/success。G8モデルSHA256 `18b39f647caa458389a1900b875337adadac74ae37e27343e7f0094ed663c492`、240局原本SHA256 `60eb9ab4dadc3776e43ac49cf27bd04137477c339850fdaa7db2265248a8d3bf` 一致。Actionsで240局の全手合法再生・12独立親を再構築し、48観測の合法手・盤面復元を検査。独立環境での240局再合法再生は未実施。
- 単一合法手走査・関数入口総時計・線形根選択は全観測でON固定。**探索木解放費用の事前予約だけ**OFF→ON→ON→OFF。同一12親（12/50/110手目各4）、同seed、200ms、各方式24観測。計測器なし、新対局・学習なし。

## 実測結果（呼出し全体）

| 指標 | 予約OFF | 予約ON |
|---|---:|---:|
| 観測数 | 24 | 24 |
| 200ms超過 | **4** | **1** |
| 中央値 | 198.069ms | 193.661ms |
| 平均 | 194.671ms | 191.013ms |
| 最大 | 207.492ms | 206.721ms |
| UCT反復合計 | 6,586 | 6,416 |
| 反復0回 | 0 | 0 |

予約ONが親ごとの2回平均で短かったのは**9/12親**。UCT反復合計は**2.58%減**。超過は4/24→1/24へ減ったが、予約ONでも50手目・親7で**206.721ms（内部205.128ms）**を観測。単一の長い反復、OS割込等の内訳はこの非計測原本だけでは確定できない。今回の予約OFFは単一走査/総時計/線形根選択がONの条件であり、以前の別実験のOFFとは同じ比較軸ではない。

## 原本と独立監査

- artifact ID **12843784911**、ZIP **2,942 bytes**、SHA256 `3118b982bf2f8a60fb756fbe2c87bb77980ee3cd83bae16f62c4f0f3dc165873`。GitHub記載digestと独立再計算一致。
- 内部48行JSONL SHA256 `b090548445d2fda1daee1e1e0caa666b7339795b01bd52f1ac1a711bdf3233a0`。ZIP内summaryの記載と一致。48行・12独立親・ABBA順・両群24件・中央値/平均/最大/超過/反復数を独立再計算一致。
- 原本artifact期限 **2026-11-27**。Google Drive長期保存は過去の `BLOCKED_FILE_REFERENCE` / file_uri型不整合が正規に解消するまで未完了。拒否を別経路で迂回しない。
- requestをmain commit `ec9f388895ceb23064a90d228501d3a5ae634411` でdisabledへ戻し、[skip run40754028467](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/40754028467) success、モデル取得・原本取得・実測・artifact工程すべてskipped。

## 判断・次工程

これは**記述統計**であり、12親×2回では統計的有意性・200ms硬上限・棋力向上・正式採用は証明できない。今後は予約ONでも残る206.721ms超過の単一反復・OS割込・探索木解放費用を、軽量計測で分離し、探索反復数との交換関係を検討する。実戦棋力は別の先後交替比較で評価。未決着を勝ち/引分教師にせず、NNUE PR#62の安全拒否も迂回しない。

### 漫画用の事実と創作素材

事実：探索木の後片付け費用を次の反復開始前に予約すると、同一12親で200ms超過が4回から1回へ減った。しかし最長206.721msが残り、主人公は「時計を守れた」とは宣言できない。セリフ案：「読む前に、片付ける時間も残す。……でも一度だけ、まだ時計を越えた。」
