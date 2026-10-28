# UCT総時計200ms・同一12局面ABBA実測結果（2026-10-29、研究のみ）

## 実行経路・固定条件

- 実装入口: [PR #71](https://github.com/mtk-ctrl/shogi-fukaji/pull/71)、branch CI [40719551452](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/40719551452)（348/348成功）→squash研究main `c7ad5191fea5c68bcd02ba1cde7218589f0d47aa`→main CI [40723569079](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/40723569079)成功。既定OFFの [40723569187](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/40723569187) は重い処理全skip。
- 27リポジトリの直近最大100runを確認し、100件超の8リポジトリは `queued/in_progress/pending/waiting/requested` を個別照合。main CIと他の既定OFF preflight終了後、起動直前にactive0・同目的重複なしを再確認。他のrunner停止なし。
- 明示request commit `5d63f331e0d23ad813747de22ad332e92bf61c18` から1runnerで [実測run 40723952839](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/40723952839) completed/success。
- G8モデルSHA256 `18b39f647caa458389a1900b875337adadac74ae37e27343e7f0094ed663c492`、240局原本SHA256 `60eb9ab4dadc3776e43ac49cf27bd04137477c339850fdaa7db2265248a8d3bf` を実行中に照合。全240局の合法再生を既存load_source/build_positions経路で実施。未決着を勝ち/引分の教師にしない。
- 単一合法手走査ON固定、総時計予算のみOFF→ON→ON→OFF、12/50/110手目各4親・計12独立親局面、各方式24観測、同一seed、200ms、深さ12。新対局・学習・正式採用なし。

## 非計測48観測の結果

| 指標 | 総時計OFF | 総時計ON |
| --- | ---: | ---: |
| 観測数 | 24 | 24 |
| 呼出し全体200ms超過 | **12/24** | **3/24** |
| 呼出し全体中央値 | 199.959ms | 198.755ms |
| 呼出し全体平均 | 196.905ms | 194.244ms |
| 呼出し全体最大 | 207.559ms | 200.359ms |
| 内部探索計測中央値 | 196.831ms | 196.694ms |
| UCT反復合計 | 5,207 | 5,137（-1.34%） |
| 0反復観測 | 0 | 0 |

同一親局面の2回平均でONが短いのは**10/12親**。ONの超過3回は50手目の親4で200.121ms、110手目の親11で200.359/200.039ms。旧方式の超過12回と比べて減ったが、**厳密200ms保証はない**。反復数も微減し、棋力向上の証明にはならない。

## 成果物と独立監査

- GitHub artifact ID `12832445665`、ZIP 2,795B、SHA256 `4dd65ce97d992fbbc4c003ac354c82ecd6069ecb6dbf0ad7641e5ab65c83d145`。GitHub API公表digestと別取得した実ZIPのSHAが一致。
- ZIP内 `abba_observations.jsonl` は48行、SHA256 `324317dadd019ed4512a865cba962eb773874ac3d5e0a8060546b1a304834075`。独立に全行を読み、4系列各12、12親SFEN同一、局面重複なし、超過数・平均・中央値・最大・反復合計・0反復数・10/12親を再集計一致。
- Actions実行ログでG8モデル/240局原本のSHA、合法再生処理、合法着手/盤面復元の検査と実行成功を確認。**独立した別環境で240局全手を再生したとは主張しない**。
- artifact保存期限は **2026-11-27**。Drive長期保存を試みたが、コネクタの参照型不整合（`file_uri` schemaはstring、実際はobject）と `BLOCKED_FILE_REFERENCE` により未保存。制限の迂回はしない。長期保存は未完了課題。
- 明示requestを `enabled:false` へ戻したcommit `e1238bd644ddc11e9fcc64b73a09d6290af9198f`。 [skip run 40724167749](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/40724167749) successでモデル取得/原本取得/実測/成果物保存がすべてskipped。

## 結論・次工程

根準備を含める総時計方式は、同一12局面の探索時間超過を12/24→3/24に減らした**研究観測**。厳密な時間上限・再現性・棋力向上は未証明。次は残る3超過の単一反復/終了処理を軽量に診断し、独立した先後交替の棋力比較を別計画とする。NNUE学習PR#62は安全拒否された書込みを再試行・迂回せず保留。雲路main、深路正式USI/既定モデル/正式採用は変更しない。

## 漫画素材（創作メモ）

「二度数えていた合法手を一度にした。それでも時間を越える。時計が根の準備を数えていなかったからだ。時計を関数入口に移した実験では超過が12回から3回へ減った。しかし最後の0.359msが、まだ『完成』と言わせない。」
