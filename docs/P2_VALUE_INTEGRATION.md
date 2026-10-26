# P2: 学習済み価値モデルの実戦探索への接続（初期実装）

## 到達点（2026-10-26）
- 深路のPython αβ探索で、評価葉にモデル推論を混合するオプションを追加。
- 現在の `value-v0.npz` は線形モデルでありNNUEではない。実戦での強化効果は未測定。
- `ValueModelFile` はnpz重みのみ読込み、形状・有限値・SHA256を確認。pickleを使わない。
- `ValueModelMix` は0〜100%で古典評価と線形value推定を混合。既定ではモデル無効。
- 学習targetは手番側勝率なので、推論スコアも手番側視点。negamaxの反転と整合させる。
- 1.0 logitを仮に100cpへ写像するが、この対応は未校正。絶対値を探索で比較するときは実測で改訂。
- 学習の結果がモデルファイルへ保存されても、既存の深路mainは自動採用しない。

## 実行例
python -m fukaji.engine
USI commands:
usi
setoption name ValueModelFile value build/model/value-v0.npz
setoption name ValueModelMix value 50
isready
position startpos
go movetime 200
quit

## 次の完了条件
1. 本物の探索自己対局で詰みが十分発生するデータ生成方法を用意し、ラベルの起点まで証明する。
2. エンジン世代を固定して学習し、モデルの参照hashをmanifestへ記録。
3. 200msでmodel ON/OFFのnodes・search depth・応手合法性を比較し、予想外の速度低下や停滞を監査。
4. 旧版との先後均衡・固定局面・事前設定局数の比較に成功する。
5. その後、小型NNUE型へモデルを置き換え、同等以上の検索速度を目指す。

## 重要な限定
今回の接続は推論経路の工学的検証であり、棋力向上を意味しない。現段階ではNNUEおよび何世代もの学習循環は未実装。