# D34：D33 NNUEを正式版へ反映

2026-11-03。オーナーの明示指示によりD33代表候補 `e0cbd2abe53b1d151bb73407f848b4087456c1466121333ad490a96aee131b0a` をFUKAJI v0.3.0として採用する。正式版の正本は `formal/model.json`。推論用重みは `models/fukaji-v0.3.0.nnue.npz`、SHA256 `146dd1b0b45bc1cda03a7b16c74f5892894f44b4e2e6e331f008439448503497`。

## 採用判断

候補の親は `c8ad43bfd17a1b6bd551cab1073f6ef13d04ad501ea4cf64f3b319a8c259e37a`、Adam step1708。3seedの先頭seed202611033301を事前に代表指定している。正式採用時点では、新100序盤×先後200局の固定比較は0局だった。オーナーの明示判断で先に採用し、この時点では棋力向上の対局証明はない。

採用後の2026-11-03に、同じ固定plan `d33-followup200` を最後まで実行した。D33代表e0対学習直前親c8、100序盤・先後200局、C++ NNUE/q0/100ms/最大512手。結果はe0が113勝・83敗・1引分・3未決着。全34,726手の合法再生と結果一致を確認した。序盤群を単位にした95%同時区間は43.17〜71.83%で50%をまたぎ、結論は「不確定」。この比較から棋力向上が証明されたとは扱わない。古い探索的対局は集計に含めていない。研究用latestは引き続き親c8/Adam1580で、正式採用と研究ポインタを分離する。

完全な研究状態は本人限定Drive file `1x6JssFUtjVldoTv5akWLs_EhlEegcffH`、SHA256 `92ab80bc0ce6f8ba976772b6ec4c9529a91ddb714fec80cc2330e51d3d6ccbd4`。Driveからの再取得SHA一致後、標準復元で別領域へ展開し、200局report・34,726手・結果・未決着・plan SHA・研究latestを再確認した。shared=false、権限はownerのみ。50%をまたぐ結果なのでD33学習goalの完了条件は満たしたが、安定強化の証明や研究latest更新の根拠にはしていない。

評価モデル・重みを公にしない保存境界は維持する。コード・manifest・集計記録のみGitHubに保存し、重み同梱パッケージと完全研究stateは本人限定Driveへ保存する。研究棋譜・Adam・全stateは正式起動パッケージに含めない。

## 実装・配布確認

実重みを使う確認では、32局面の固定深さ2結果一致、USI2局・合法60手、stop応答を確認した。探索応答時間は先行する5回のパッケージ測定で36.335〜308.899ms、後続の独立bundle確認で28.429msだった（計6観測、movetime 20ms要求）。開局から20手だけの診断では最大22.29msだったが、後半局面・深い探索の網羅試験ではない。最終Drive復元確認では25.367ms。測定値のばらつきが大きく、20ms要求や通常設定100msの厳密な上限を満たすとは確認できていない。これは実装・配布確認であり棋力測定ではない。

`formal/model.json` のSHAを正式USIで検証し、欠損・破損時に起動停止する。従来のC++ NNUE、履歴処理、stop、時計配分、旧版起動経路を保持する。D32のv0.2.0配布物とlegacy v0.1.0は履歴として残す。

正式パッケージ生成：`python scripts/build_formal_bundle.py --out build/fukaji-v0.3.0.tar.gz --source-commit COMMIT`。manifestが指す重みと全メンバーのSHAを `BUNDLE.json` に記録する。展開・依存導入後、`python scripts/verify_formal_release.py --out build/formal-real-model-verification.json` で実モデルの起動、32局面の深さ2一致、2局・合法USI、stop、bundle hashを検証する。これは配布・起動の正しさの確認であり、棋力比較ではない。

GitHub CI、本人限定Drive配布、再取得SHA、別ディレクトリ展開と実モデル起動の証跡は、`research/releases/v0.3.0/verification.json` およびAddnessの深路Goalへ保存する。

旧版復元記録：[D32 v0.2.0](D32_FORMAL_NNUE_RELEASE_20261102.md)。採用判断：[D33採用決定](decisions/2026-11-03_D33_NNUE正式採用.md)。
