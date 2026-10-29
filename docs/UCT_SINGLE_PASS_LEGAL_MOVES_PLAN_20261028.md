# 深路G8：UCT合法手リスト二重走査の検証計画（2026-10-28）

## 目的・根拠
前段の固定G8/240局原本・親6/11の119反復では、展開758.682ms中 `list(board.legal_moves)` が747.509ms（98.527%）を占めた。これは展開内の比率であり、全探索時間や改善可能割合ではない。

python-shogiの `LegalMoveGenerator.__len__` は `board.generate_legal_moves()` を最後まで列挙して数える。同クラスの `__iter__` も `board.generate_legal_moves()` を返す。Pythonの `list(generator)` は長さ取得後に反復するため、**合法手生成器を2回走査する構造**になっている。 `generate_legal_moves` は疑似合法手を生成し、各候補を `is_suicide_or_check_by_dropping_pawn` で判定する。判定はpush/popを伴う。

一次ソース： https://github.com/gunyarakun/python-shogi/blob/a814cbff51b0703df2503ee59408fefd2e464ae5/shogi/__init__.py （LegalMoveGenerator / Board.generate_legal_moves / Board.is_suicide_or_check_by_dropping_pawn）。現行依存の固定 `python-shogi==1.1.1` とコードの一致はCI環境で確認すること。

## 研究候補
研究専用の `single_pass_legal_moves=True` のときだけ、根と展開の `list(board.legal_moves)` を `[move for move in board.legal_moves]` に変更する。既定はfalse。棋力・探索戦略・評価モデル・既定USIは変更しない。比較は単一の材料化方式の違いだけにする。

## 検証ゲート
1. branch CIで、`LegalMoveGenerator.__len__` を失敗させても単一走査が動くこと、固定seed/固定反復の選択手・反復数が一致することを検査。
2. 初期・王手中・持駒・打ち歩詰禁止・詰み終局・4回目千日手局面で合法手の集合**と順序**、着手合法性、盤面と履歴の復元が一致すること。
3. 研究main統合後main CI成功。重い実測前に27repoのqueued/in_progress/pending/waiting/requestedと同目的重複を確認し、別明示requestで1runner診断する。
4. G8モデルSHA256 `18b39f647caa458389a1900b875337adadac74ae37e27343e7f0094ed663c492` と原本240局JSONL SHA256 `60eb9ab4dadc3776e43ac49cf27bd04137477c339850fdaa7db2265248a8d3bf` を固定し、全240局合法再生と親6/11・同一12局面の非計測ABBAで時間と反復を測る。SHA・結果を独立監査し、requestをdisabledへ戻してskipを確認。
5. 200msの厳密保証、実戦の強さ、正式採用は別の判断であり、今回のCI成功のみで主張しない。

## 未実施
本計画の記載時点では、branch CI・main CI・実局面での速度比較・棋力比較は未完了。NNUE学習PR#62は安全拒否された書込みを再試行せず未統合のまま。
