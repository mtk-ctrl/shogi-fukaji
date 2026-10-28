# UCT探索木解放費用予約・固定12親ABBA計画（2026-10-29）

研究専用・既定OFF。PR#77の事前予約だけをOFF→ON→ON→OFFに切替え、同一12親・同一seed・200ms・非計測48観測を1runnerで比較する。単一合法手走査、関数入口からの総時計、線形根選択は全観測でON固定。雲路main/深路正式USI/G8モデルは変更せず、正式採用はしない。

## 実行前の必須順序
branch CI成功→squash研究main→main CI成功→所有27repoのqueued/in_progress/pending/waiting/requestedと同目的重複の全調査→空きなら別明示requestをenabled=trueへ更新→実run→監査→request disabled/skip。workflowはmain限定、scheduleなし、read-only、他runを停止しない。

## 固定データ・監査
G8モデルSHA256 `18b39f647caa458389a1900b875337adadac74ae37e27343e7f0094ed663c492`、240局原本SHA256 `60eb9ab4dadc3776e43ac49cf27bd04137477c339850fdaa7db2265248a8d3bf`。全240局合法再生後に12/50/110手目各4独立親を構築し、各呼出しの合法手・盤面復元・反復数・内外時間を監査。0反復でも合法手なら記録し、未決着を勝ち/引分教師にしない。48行原本・統計・artifact SHAを独立照合する。

## 解釈と終了
超過率・最大/中央値・反復数は記述統計。12親×各2回は有意差・棋力改善・200ms硬上限保証の証明ではない。研究結果はGitHub技術正本とAddness本文/AIメモリへ差分保存・読戻し。DriveのBLOCKED_FILE_REFERENCE等を迂回しない。
