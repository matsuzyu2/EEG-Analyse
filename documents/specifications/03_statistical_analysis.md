# 03_statistical_analysis.md — 統計解析仕様（within-subject）

対象スクリプト: [src/04_condition_statistics.py](../../src/04_condition_statistics.py)

作成日: 2026-01-11

## 1. 目的
Good Responders に限定して、被験者内差分

- TFR: `Target - Control`（freq×time）
- HEP: `Target - Control`（time）

に対し、**1標本クラスター置換検定**（`permutation_cluster_1samp_test`）を実施し、
有意クラスターを可視化・保存する。

本モジュールは **中間ファイルの読み込みのみ**を行い、TFR/HEPの再計算は行わない。

## 2. 入力
### 2.1 Good Responders リスト
- 既定: `data/classification/good_responders.csv`
- 必須列: `session_id`
- `session_id` は `subject_id` として扱う（前処理/中間生成物のディレクトリ名と一致する前提）

### 2.2 中間生成物（被験者ごと）
被験者 `sid` について、以下4ファイルが揃っている必要がある。

- TFR: `data/processed/<sid>/<sid>_tfr_target.h5`
- TFR: `data/processed/<sid>/<sid>_tfr_control.h5`
- HEP: `data/processed/<sid>/<sid>_hep_target-ave.fif`
- HEP: `data/processed/<sid>/<sid>_hep_control-ave.fif`

欠損がある被験者は警告ログの上でスキップする。

## 3. 被験者抽出フロー（データ存在確認）
1. `good_responders.csv` から `session_id` を読み取り（順序保持・重複除去）
2. `data/processed/<sid>/` の存在確認
3. 上記4 artifact の存在確認
4. 揃った被験者のみを解析対象とする

制約:
- 解析対象が2名未満の場合は例外で停止（統計が成立しないため）

## 4. ROI定義（辞書定義とチャンネル構成）
ROIは本スクリプト内で辞書として定義する（探索的・柔軟にするため）。

定義（候補）:
- `Frontal`: `["Fz", "F1", "F2", "F3", "F4", "FCz", "FC1", "FC2", "Cz"]`
- `Visual`: `["Oz", "O1", "O2"]`
- `Parietal`: `["Pz", "P3", "P4", "CPz", "CP1", "CP2"]`

備考:
- ROIの定義は「候補チャンネル」のリストとして持ち、被験者ごとのデータに存在しないチャンネルは自動的に除外して平均する（存在chのみで平均）。
- within-subject（Target/Control差分）のため、ROI平均に用いるチャンネルは **Target/Controlの両方に存在するもの（共通部分）** に限定する。
  - これにより、被験者内差分が「同一チャンネル集合上の差分」になることを保証する。
- 共通部分が0チャンネルの場合は当該被験者をスキップし、全体で有効被験者が2名未満になった場合は例外で停止する。

## 5. データ整形（ROI平均）
### 5.1 TFR（AverageTFR）
- 読み込み: `mne.time_frequency.read_tfrs(path)`
  - MNEのAPI差分により、戻り値が `list[AverageTFR]` または `AverageTFR` の場合があるため、両方を吸収する
- ROI平均: ROI候補チャンネルのうち、Target/Controlに共通して存在するチャンネルのみを `pick` → `data.mean(axis=0)`
- 形状:
  - 被験者ごとの差分 `X_tfr`: `(n_subjects, n_freqs, n_times)`

### 5.2 HEP（Evoked）
- 読み込み: `mne.read_evokeds(path, condition=0)`
  - 戻り値が `list[Evoked]` または `Evoked` の差分を吸収
- ROI平均: ROI候補チャンネルのうち、Target/Controlに共通して存在するチャンネルのみを `pick` → `data.mean(axis=0)`
- 形状:
  - 被験者ごとの差分 `X_hep`: `(n_subjects, n_times)`

## 6. 統計手法（permutation_cluster_1samp_test）
### 6.1 共通（差分に対する1標本検定）
検定対象:
- `X = Target - Control`

TFR/HEPともに以下を実行する。
- 関数: `mne.stats.permutation_cluster_1samp_test`
- `adjacency=None`（隣接行列は使わない）
- `out_type="mask"`（クラスターをboolマスクで受け取る）

### 6.2 クラスター形成閾値
クラスター形成は **t分布の臨界値**に基づく。

- 被験者数 `n_subjects` → `df = n_subjects - 1`
- `cluster_alpha` と `tail` から t閾値を決める:
  - `tail=0`（両側）: `t.ppf(1 - cluster_alpha/2, df)`
  - `tail=1`（右側）: `t.ppf(1 - cluster_alpha, df)`
  - `tail=-1`（左側）: `t.ppf(cluster_alpha, df)`

### 6.3 デフォルトパラメータ（仕様として固定）
- `n_permutations = 2000`
- `cluster_alpha = 0.05`
- `alpha = 0.05`（クラスターp値の有意判定）
- `tail = 0`（両側）
- `seed = 0`

有意マスク生成:
- `cluster_p < alpha` のクラスターmaskをOR結合して `sig_mask` を作る

## 7. 可視化仕様
### 7.1 TFR図（ROIごと: 3パネル）
出力: `tfr_<ROI>_target_control_diff.png`

内容:
1. Target平均（freq×time）
2. Control平均（freq×time）
3. 差分平均（Target-Control）

描画仕様:
- x軸: Time (s)
- y軸: Frequency (Hz)
- time=0 に縦線
- 差分パネルは発散カラーマップ `RdBu_r`
  - 表示レンジは `abs(diff_mean)` の 98パーセンタイルで対称スケーリング（0の場合は最大値へフォールバック）
- 有意クラスターは差分パネルに `contour`（黒線）で重畳

### 7.2 HEP図（ROIごと: 重ね描画 + SEM + 有意バー）
出力: `hep_<ROI>_target_control.png`

内容:
- Target平均波形（赤） + SEMシェーディング
- Control平均波形（青） + SEMシェーディング
- time=0 に縦線
- 有意区間は図の下部にバー表示（連続区間を自動でまとめて水平線）

## 8. 出力
### 8.1 出力ディレクトリ
- 既定: `data/group_analysis_within_subject/`

### 8.2 生成物
ROIごとに以下を生成:
- `tfr_<ROI>_target_control_diff.png`
- `hep_<ROI>_target_control.png`
- `stats_<ROI>.npz`

NPZの主な保存キー（代表）:
- ROIメタ:
  - `roi`
  - `roi_candidates`（候補チャンネル）
  - `tfr_roi_channels`（TFRで実際に使われたチャンネル）
  - `hep_roi_channels`（HEPで実際に使われたチャンネル）
  - `tfr_subjects`, `hep_subjects`（実際に解析に使われた被験者）
  - `tfr_roi_pick_mask`, `hep_roi_pick_mask`（(n_subjects, n_candidates) のbool）
- 検定出力:
  - `tfr_cluster_p_values`, `tfr_clusters`, `tfr_sig_mask`
  - `hep_cluster_p_values`, `hep_clusters`, `hep_sig_mask`

## 9. CLI仕様（主要）
デフォルト:
- `--good_csv data/classification/good_responders.csv`
- `--out_dir data/group_analysis_within_subject/`
- `--roi Frontal`

主要オプション:
- `--roi`（複数指定可）
- `--n_permutations`, `--cluster_alpha`, `--alpha`, `--tail`, `--seed`
