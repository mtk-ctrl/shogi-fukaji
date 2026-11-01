# D19：研究用NNUE探索での対局履歴再利用

研究用自己対局の探索準備から、盤面複製と過去の全着手の逆再生を省く。対局側が維持する局面hashと王手履歴だけを検索用Positionへコピーする。NNUEの累積値・undoは共有せず、手番側のモデルから作り直す。

- `select(..., history=position)` は同じ盤面・履歴長・現在hashを検査する。省略時は従来の独立履歴再構築を維持する。
- `online_cycle.play(..., reuse_history=True)` は研究用の履歴再利用。先後別の2個の真偽値で方式だけを変える比較も可能。各対局の条件へ保存する。
- 千日手・連続王手の負け・詰み判定は元の規則を使う。別モデルの累積値を再利用しない。
- `tests/test_online_history.py` は別モデルの固定深さ一致、履歴と累積値の分離、通常千日手、連続王手、古い履歴拒否、時間中断後の復元、先後別条件を検査する。
- `scripts/d19_history_benchmark.py` は事前固定したtrain局面のABBA比較。固定深さの着手・評価・節点一致と、指定時間での完了深さ・節点・呼出全体時間を分ける。
- `scripts/d19_history_arena.py` は凍結NNUE・新しい先後ペア・同じ指定時間で履歴再利用だけを変える有限monitor比較。先に親群を保護し、途中失敗は学習pauseを残す。

正式USI・G8・配布モデル・比較基準は変更しない。研究用の実装と、実際の棋力向上の実証・正式採用を区別する。思考時間の厳密な上限保証は追加していない。

この文書とコードには、回収した学習モデル・完全state・実対局棋譜を同梱しない。それらの公開保存とは分離したコード変更である。

```sh
python -m unittest discover -s tests -p 'test_online*.py' -v
python scripts/check_fukaji_contracts.py
```
