# 深路（ふかじ） / FUKAJI

雲路 KUMOJI と独立した、自己対局と価値学習による将棋AI研究プロジェクト。

## 現在の実装
- `fukaji/selfplay.py`: python-shogiの合法手を使うランダム基準器の自己対局生成（詰み以外を勝ちとして扱わない）
- `fukaji/learning.py`: JSONL棋譜の全着手検証、詰みと勝者照合、重複排除、対局単位の分割、NumPy線形valueモデル訓練
- `tests/`: 特徴量、学習、違法手拒否の単体検査
- `.github/workflows/learning-smoke.yml`: テストはpushごと。手動起動で自己対局→学習→モデル成果物を保存

**制約**: value-v0はNNUEではなく、モデル推論はまだ対局エンジンに接続していない。生成器も雲路エンジンではない。棋力向上は未検証。外部AIの評価値や最善手は使わない。

## 実行
Actions → FUKAJI learning pipeline → Run workflow（初期attempts=40, epochs=4）。完了した詰み棋譜がないときは失敗させる。学習済みモデルはActions artifactで取得。

ローカル:
```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m fukaji.selfplay --attempts 40 --out build/selfplay.jsonl
python -m fukaji.learning --games build/selfplay.jsonl --out build/model
```

入力は1行1局のJSONL、キーは `start_sfen`, `moves`（USI手の配列）, `winner`（先手0、後手1、未決着null）, `source`（selfplay）の4つ。勝敗付きの入力は詰み終了を検証する。未終局・出所不明・非合法手を教師データにしない。

## 今後の必須工程
1. 雲路の特定commitのルール層・USI・探索を、GPL条件と出所表示を維持して必要部分だけ独立移植
2. 真正な雲路由来の自作探索による深路の自己対局を学習の入口に接続
3. 小型NNUE型モデルを実装・学習し、対局時の評価関数として推論接続
4. 分岐局面の研究、未使用対局の評価と旧版・雲路比較を自動化

新AIのコード、Actions、学習重み、研究世代は元リポジトリへ影響させない。
