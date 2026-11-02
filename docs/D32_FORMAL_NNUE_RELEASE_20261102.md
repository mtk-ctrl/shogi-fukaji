# D32：D30 NNUEを正式版へ反映

2026-11-02。本人指示による正式採用。詳細は [現行仕様](現行エンジン仕様.md) と [判断記録](decisions/2026-11-02_D30_NNUE正式採用.md)。

`formal/model.json` を正本としてNNUEを固定する。研究snapshot912db54fから推論用重み3705c31fへ全テンソル一致で変換した。正式USIにNNUE読込、C++探索、全履歴、atomic stop、再対局、時間設定を実装した。

## 検証と配布

全回帰 **701検査成功（90.253秒）**、正式契約・相対リンク・保存状態検査成功。重点20検査を含み、公開CIでは合成NNUEを用いて実装を検査する。学習済み重みは公開CIへ送らない。

採用した実モデルで初期・履歴付き・持駒・王手の32局面を深さ2で照合し、従来C APIと停止対応C APIの着手・評価値・節点・評価回数・深さが全一致。USI実プロセスで2局・計60手の連続合法着手、モデルSHA表示、履歴の伸長、新しい局面系列、再対局、isready中の探索、stop、旧探索結果の破棄を確認した。20ms指定の最大観測応答22.971ms、stop応答0.927ms。これらは起動・組込みの正しさの検査で、棋力を推定する60手ではない。

同梱パッケージを別ディレクトリへ展開し、全155ファイルのSHA照合、C++再ビルド、同じ32局面の一致とUSI60手を再確認した（最大22.084ms、stop0.632ms）。正式保存物もアップロード後に再取得し、SHAと復元起動を確認する。

`python scripts/build_formal_bundle.py --out build/fukaji-v0.2.0.tar.gz --source-commit COMMIT` でコード・仕様・推論用重みのパッケージを作る。重みSHAの検査に成功しなければ生成しない。全メンバーのSHAはBUNDLE.jsonで確認できる。研究state/Adam/棋譜は同梱しない。展開後requirements.txtをインストールしrun-fukaji.shを起動する。

成果物は本人限定Driveへ保存し、メタデータ・再取得SHA・展開後起動を検証する。公開GitHubには学習済み重みを保存しない。正式コードの採用経路と旧版の再現経路を分け、旧workflow/検査は--legacyを明示する。

旧正式コード364b2645797d436e531a28b0b5013a6cc70bfbe1、旧仕様と旧formal-contract-v010.json、固定benchmarks/baseline_ref.txt、研究latest/Adam/15周巻戻し先を保全。

Goal: https://www.addness.com/goals/ca331204-b4c0-4c34-ac56-f0a25d933632
