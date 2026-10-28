# 深路G8 UCT反復段階別診断：実行入口（2026-10-28）

- 研究専用・既定OFF。PR#64の任意traceを使い、既存の同一12局面ABBAで超過が残った**親6・親11**だけを調べる。G8モデルSHA `18b39f647caa458389a1900b875337adadac74ae37e27343e7f0094ed663c492` と240局原本JSONL SHA `60eb9ab4dadc3776e43ac49cf27bd04137477c339850fdaa7db2265248a8d3bf` を固定。
- 出所はG8学習run40636602223と独立240局run40637486947。元の240局を全手合法再生し、既存 `build_positions` の12親のうち6・11を同じ履歴で復元。各2回200ms、seed固定、選択手の合法性、盤面・履歴復元を検査。
- 各反復の選択/展開/葉評価/逆伝播/undo/総時間・予約時間・残時間、各局面の50/90/99パーセンタイル・最大・予測予約超過件数を原本JSONLに保存。統計は原因の仮説生成に使い、**計測負荷があるため前回の非計測ABBAの200ms経過時間とは直接比較しない**。
- workflow `uct-phase-profile.yml` はmain限定・読取権限・scheduleなし・concurrency取消なし。明示requestは既定OFF。PR→branch CI→squash研究main→main CI→27repoの5状態・同目的重複確認→別commitでrequest enabled→実run→全ZIP SHA/統計監査→disabled/skip。
- **実行・結果は別工程**。未決着は教師化しない。正式USI・雲路main・NNUE学習PR#62は変更しない。
- 物語素材（創作案、本人発言ではない）：「同じ2局面で何度も超過する。何が遅いか決めつけず、1反復を5つに分けて測り直す」。
