# 深路 G8：実詰みのみの凍結学習・研究候補生成（2026-10-28）

**G8研究モデルの生成まで実行済み。棋力向上は未証明、正式採用なし。**

## 正本と実行
- 研究基盤[PR #50](https://github.com/mtk-ctrl/shogi-fukaji/pull/50)、squash main `ce4708f4e1c6d4bc6a0fb5bedc92e531a409639f`。branch CI run40636115349で274テスト成功。main CI run40636375283 preflight成功・train skipped、他main CI run40636375310/40636375287/40636375339成功。
- 所有27repoのqueued/in_progress/pending/waiting/requestedを確認し全0。その後requestを別commit `070bbfa86ea868a159e9092eb80d14f0a3e9be52` で明示有効化。
- [本実行run40636602223](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/40636602223)：preflight success、train success、成果物upload success。
- 凍結親G7モデル SHA256 `417d18e7b64494c9febdf6d217afbe4e4979e68d04528f16db8b45c0bcde2a0e`。
- **生成したG8研究候補モデルSHA256 `18b39f647caa458389a1900b875337adadac74ae37e27343e7f0094ed663c492`**。
- 研究artifact ID `12793956048`、GitHub ZIP digest `sha256:9b5c004446f4b93a529d52fa96243ccbd5ac434373fe1ff69191bee9c8e6306e`（GitHubメタデータ・CIログ確認）。**別環境でのZIP実ダウンロード/SHA独立照合はこの段階では未実施**。次の240局比較preflightで元ZIPバイトSHAと中身を照合する設計。

## 学習・保護分割
| 種類 | 実詰み局 | 備考 |
|---|---:|---|
| 学習 | **665** | 旧459＋新206のみ |
| 検証 | 81 | 学習・調整に不使用 |
| テスト | 84 | 学習・調整に不使用 |
| 合計 | 830 | 全手合法再生 |
| 未決着 | 142 | 勝ち・引分・教師なし |

旧557親群と新400親群の割当を保持、合計957独立親群。旧train447/validation55/test55、新train320/validation40/test40。新実詰みtrain206/validation26/test26。旧・新親群の交差重複0。長考変更40局面は教師化せず、外部AI評価も使用していない。

事前固定optimizer：8epoch、学習率0.02、seed42、親G7へのL2係数0.001、更新量縮小0.5。学習済み係数の変化1463個。保留群は勾配・学習率・epoch・候補選定に使っていない。

## 学習後の勝敗予測損失（小さいほど良い）
| 分割 | G7 | G8 |
|---|---:|---:|
| 学習665局 | 0.549368 | **0.537405** |
| 検証81局 | 0.605587 | **0.593355** |
| テスト84局 | 0.544105 | **0.537034** |

これは自己対局の結果ラベルへの予測適合度であり、**G8が実戦でG7より強い証明ではない**。保留群の数値を見てパラメータを変更しない。独立seed先後交替240局の実戦で棋力を検証する。

学習requestはcommit `0ceeb849e955893bb73d76896d9267697177ee43` でdisabledへ戻した。skip CIの成功は別途確認する。雲路正式版の変更なし、研究モデルの正式採用なし、Goal未完。
