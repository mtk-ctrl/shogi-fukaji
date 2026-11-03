# D34：D33 NNUEを正式版へ反映

2026-11-03。オーナーの明示指示によりD33代表候補 `e0cbd2abe53b1d151bb73407f848b4087456c1466121333ad490a96aee131b0a` をFUKAJI v0.3.0として採用する。正式版の正本は `formal/model.json`。推論用重みは `models/fukaji-v0.3.0.nnue.npz`、SHA256 `146dd1b0b45bc1cda03a7b16c74f5892894f44b4e2e6e331f008439448503497`。

## 採用判断

候補の親は `c8ad43bfd17a1b6bd551cab1073f6ef13d04ad501ea4cf64f3b319a8c259e37a`、Adam step1708。3seedの先頭seed202611033301を事前に代表指定している。正式採用時点では、新100序盤×先後200局の固定比較は0局で未実施。オーナーの明示判断による採用であり、棋力向上の対局証明はない。固定比較は元の条件で継続する。

評価モデル・重みを公にしない保存境界は維持する。コード・manifest・集計記録のみGitHubに保存し、重み同梱パッケージは本人限定Driveへ保存する。研究棋譜・Adam・全stateは正式パッケージに含めない。

## 実装・配布確認

実重みを使う確認では、32局面の固定深さ2結果一致、USI2局・合法60手、stop応答を確認した。探索応答時間は、先行する5回のパッケージ測定で36.335〜308.899ms、後続の独立bundle確認で28.429msを観測した（計6観測、movetime 20ms要求）。開局から20手だけの診断では最大22.29msだったが、後半局面・深い探索の網羅試験ではない。測定値のばらつきが大きく、20ms要求や通常設定100msの厳密な上限を満たすとは確認できていない。今回の最終Drive復元確認の値はrelease verification記録へ別記する。これらは実装・配布確認であり棋力測定ではない。

`formal/model.json` のSHAを正式USIで検証し、欠損・破損時に起動停止する。従来のC++ NNUE、履歴処理、stop、時計配分、旧版起動経路を保持する。D32のv0.2.0配布物とlegacy v0.1.0は履歴として残す。

正式パッケージ生成：`python scripts/build_formal_bundle.py --out build/fukaji-v0.3.0.tar.gz --source-commit COMMIT`。manifestが指す重みと全メンバーのSHAを `BUNDLE.json` に記録する。展開・依存導入後、`python scripts/verify_formal_release.py --out build/formal-real-model-verification.json` で実モデルの起動、32局面の深さ2一致、2局・合法USI、stop、bundle hashを検証する。これは配布・起動の正しさの確認であり、棋力比較ではない。

GitHub CI、本人限定Drive配布、再取得SHA、別ディレクトリ展開と実モデル起動の証跡は、`research/releases/v0.3.0/verification.json` およびAddnessの深路Goalへ保存する。

旧版復元記録：[D32 v0.2.0](D32_FORMAL_NNUE_RELEASE_20261102.md)。採用判断：[D33採用決定](decisions/2026-11-03_D33_NNUE正式採用.md)。
