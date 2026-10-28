# UCT凍結保持群12親・後片付け予約ABBA入口（2026-10-29）

状態：**研究用入口の設計・実装候補。実測前は未検証**。保持群を時間予約係数の調整に使用しない。正式エンジン・雲路main・G8モデル・既定設定・学習は変更しない。

## 固定条件
- 保持群は [凍結JSON](./UCT_HOLDOUT_12_PARENTS_20261029.json) の pair12〜23、12/50/110手目各4。開発群pair0〜11を一切混ぜない。原本240局を全手合法再生してから選ぶ。
- G8モデルSHA256 `18b39f647caa458389a1900b875337adadac74ae37e27343e7f0094ed663c492`、原本240行SHA256 `60eb9ab4dadc3776e43ac49cf27bd04137477c339850fdaa7db2265248a8d3bf` の**実バイト**照合。
- 同一12親・各親同seed・200ms、OFF→ON→ON→OFFの48観測。単一合法手走査ON、総呼出時計ON、線形根選択ONを固定し、探索木解放の事前予約だけ切替。探索木の解放を遅らせない。
- `fukaji.uct_cleanup_reserve_abba` のABBA順序・統計要約を再利用し、`fukaji.uct_holdout_freeze` の240局合法再生と `uct_generalization_audit` の保持群48行検査を追加。合法手・盤面/履歴復元・出所・SFEN・seedを照合。0反復でも合法手なら正当な観測。
- `.github/fukaji-uct-holdout-reserve-abba-request.json` は既定 `enabled:false`。新main CI成功→所有27repoのqueued/in_progress/pending/waiting/requestedと同目的重複を全調査→空きなら**別明示request commit**で1runner。実行後はdisabledへ戻し、後続runで重い工程skipを確認。

## 結果の判断と監査
- 48観測原本・集計・ZIPのSHA256を独立再計算。24対24の超過率・中央値・反復数・12親別対応差を記録。Actions上の全240局合法再生と独立環境の再生を区別する。
- 開発群の4/24→1/24は過去の探索的結果であり、保持群に転記しない。保持群は独立したゲーム親だが、同一run・同一環境・同一親反復であり、無条件に独立同分布の試行とみなさない。
- 非割込みの長い反復、OS停止、探索木解放があるため200msの**硬い保証ではない**。着手差・反復数は棋力向上を証明しない。未決着に勝敗/引分教師を付けない。係数再調整・正式採用はオーナー判断なしに行わない。
- 研究成果はGitHub技術正本とAddness Goalへ差分保存・読戻し。Drive原本ZIPのBLOCKED_FILE_REFERENCE等は別経路で迂回しない。

漫画素材：「十二の新しい盤面を封印した。速さを確かめるためであって、答えに合わせて時計をいじるためではない。」
