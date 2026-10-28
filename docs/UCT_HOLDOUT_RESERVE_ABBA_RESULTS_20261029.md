# UCT探索木解放予約：凍結保持群12親の独立比較結果（2026-10-29）

状態：**研究用保持群での非計測ABBA48観測・原本監査まで完了**。開発群の係数を保持群で調整せず、そのまま固定して評価。**200ms厳密保証・棋力向上・正式採用は未証明**。雲路main・深路正式USI/G8モデル/既定値不変更。

## 手順・実行証拠
- 入口[PR#87](https://github.com/mtk-ctrl/shogi-fukaji/pull/87) branch CI [40784002005](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/40784002005) 391/391成功→研究main `18c8517c8ad9e4a053876cab4a9cbaf46a84cded`、main軽量CI [40784175647](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/40784175647) 成功。ただしmainの研究workflow [40784175659](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/40784175659) は、**別リポジトリにのみ存在する事前検査スクリプトを参照したため失敗**。重い実験は未開始。
- [PR#88](https://github.com/mtk-ctrl/shogi-fukaji/pull/88) で深路に存在する検査へ修正し、誤参照防止テストを追加。branch CI [40784312106](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/40784312106) 392/392成功→研究main `dfe914a5dad13f70266c3e2d3c936f2dfd9179cd`→main CI [40784543103](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/40784543103) 成功。
- [PR#89](https://github.com/mtk-ctrl/shogi-fukaji/pull/89) で「メタデータ検査のみのlegality_replayed:false」を、**実際の240局全手と保持群12親の合法再生が完了した後にだけtrueにする**証跡検査を追加。branch CI [40784722450](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/40784722450) 393/393成功→研究main `e3207c5b03f3e0c405998cbb37b200c2b927933d`→main CI [40784930135](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/40784930135) 成功。mainの既定OFFコード・テスト・requestを読み戻した。
- 重いrun前に所有27リポジトリの最新最大100runを確認し、100件超の8リポジトリではqueued/in_progress/pending/waiting/requestedの5状態を個別に再検索。**有効run 0**、同目的の未完了run/Issue/PR重複なし。別目的runを停止せず、1runnerの別明示request commit `a27ca5ee815c573a7f99c658c9485aa39bb619fe` で [実run40785258037](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/40785258037) completed/success。
- 固定G8モデルSHA256 `18b39f647caa458389a1900b875337adadac74ae37e27343e7f0094ed663c492`、元240局JSONL SHA256 `60eb9ab4dadc3776e43ac49cf27bd04137477c339850fdaa7db2265248a8d3bf` の実バイト照合。Actions上で240局全手と保持群12親を合法再生。開発群pair0〜11と重複しない保持群pair12〜23、12/50/110手目各4、同一seed、200ms、単一走査・総時計・線形根選択ON固定。**予約のみOFF→ON→ON→OFF**、各方式24観測。

## 保持群の実測値

| 指標 | 予約OFF | 予約ON |
|---|---:|---:|
| 200ms超過 | 5/24 | 1/24 |
| 呼出時間中央値 | 198.478ms | 194.418ms |
| 呼出時間平均 | 194.746ms | 190.524ms |
| 最大 | 201.878ms | 200.677ms |
| 内部時間中央値 | 196.041ms | 192.133ms |
| UCT反復合計 | 5,795 | 5,910 |
| 反復0件 | 0 | 0 |

- 超過回数は5→1（80%減）、中央値は4.060ms短縮、平均は4.222ms短縮。**ONの方が短い親9/12**。反復数は+115（約+1.98%）だが、時間ゆらぎを含み、探索効率/棋力改善と断定しない。
- ONでも**110手目pair23**で外部200.677ms・内部199.854ms・59反復の超過が1件残った。従って硬い200ms上限は満たさない。
- 開発群の別実測はOFF4/24→ON1/24、中央値198.069→193.661ms、反復6586→6416。保持群で同方向の超過減少・時間短縮が見られたが、保持群は独立ゲーム親でも同一CI環境・同一親の反復であり、厳密な独立標本ではない。**予約係数は変更していない**。

## 独立成果物監査・限界
- 実run artifact ID `12857959815`、ZIP **3,696 bytes**、SHA256 `f0bbbb95ea02718aa63a0613d65e552dde2380368f8de35c23582319c273a05b`（GitHubのartifact digestとも一致）。内部 `holdout_observations.jsonl` **48行**、SHA256 `32982d7c81c646b316644cb4bc42bdfb6904a6921817e0671fd7e94ee03de0dd`。ZIPを独立取得して再計算一致。12親/4段階のABBA順序・手番分布・合法/盤面復元フラグ・超過・中央値・平均・最大・反復数・9/12対応差を独立再計算一致。全48行合法・盤面復元true、開発群混入0。
- 独立ローカルでの**240局全手再合法再生は未実施**（Actionsでの全手再生は成功）。48行の合法/盤面復元は実行時に検査し、独立側では記録の一致を監査。CI runを独立実行環境の証明としない。統計的有意性・棋力向上・OS停止を含む厳密200ms保証は未証明。
- 実験依頼は commit `875987539c4bc325aa25f1057024e9b7f109303d` でdisabledに戻し、[skip run40785403089](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/40785403089) completed/success、モデル取得・原本取得・探索・成果物生成の4工程が全skipped。別目的run停止なし。
- Artifact期限は2026-11-27T23:09Z（JST 11/28 08:09頃）。Driveへの原本ZIP長期保存は既存のBLOCKED_FILE_REFERENCE等の正規解消まで保留、迂回しない。NNUE学習PR#62の安全拒否も迂回しない。

## 次の判断
時間超過削減は保持群でも再現傾向があるが、**超過1件が残る**。200msを厳密に守るには非割込みの長い反復・OS停止・探索木解放を含む設計上の限界を別途検討する。棋力は独立先後交替対局で別に評価し、未決着を勝ち/引分教師にしない。正式採用・係数調整はオーナー判断なしに行わない。

漫画素材：「別の十二局面でも、時計を越える回数は減った。でも最後の一度が、まだ残っている。」
