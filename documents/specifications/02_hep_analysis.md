# 02_hep_analysis.md — HEP解析仕様

対象スクリプト: [src/03_analysis_hep.py](../../src/03_analysis_hep.py)

作成日: 2026-01-11

## 1. 目的
各被験者のセッションEEGと、対応するECG R波ピーク系列（JSON）から HEP（Heartbeat-Evoked Potential）を算出し、
セッション種別に基づく Target/Control 条件で **Evoked を中間保存**する。

本モジュールは「計算」担当であり、群レベル統計は行わない（統計は04）。

## 2. 入力
### 2.1 前処理済みEEG（セッション単位）
- `data/processed/<subject_id>/<subject_id>_sess01_clean.fif`
- `data/processed/<subject_id>/<subject_id>_sess02_clean.fif`

※バッチモードでは `data/processed/<id>/` 配下の `*_clean.fif` を列挙する。

### 2.2 ECGピーク（R波）JSON
- バッチモードの既定ディレクトリ: `data/ecg/<subject_id>/`
- このプロジェクトでは多様なピークJSON（baseline/rest/task等）が存在しうるため、HEPでは**セッション用のみ**を自動選別する。

自動選別ルール:
- `*Session_01*.json`（または `Session_1`）を **1つだけ**
- `*Session_02*.json`（または `Session_2`）を **1つだけ**

両方が一意に取れない場合はエラー停止する。

JSON仕様:
- キー `peak_indices` が必須
- `peak_indices` は 1次元配列（空は不可）

### 2.3 条件マニフェスト（Target/Control分類）
- `data/processed/<subject_id>/conditions_manifest.json`
- `session_condition_map` を参照し、`sess01`/`sess02` を Target/Control に分類する

分類規則:
- Target: `increase`, `decrease`, `inc`, `dec`
- Control: `control`

## 3. 条件マッピング（Session_01/02インデックスによる紐付け）
本スクリプトの前提は、

- EEG側: `..._sess01_clean.fif` / `..._sess02_clean.fif`
- ECG側: `...Session_01....json` / `...Session_02....json`

が **同一セッション番号で1:1対応**すること。

実装上は、
- ECG JSONを `Session_01`/`Session_02` で選別
- EEG FIFから `sess01`/`sess02` をファイル名から抽出
- `conditions_manifest.json` の `session_condition_map["sessXX"]` により条件を決める

という形で、セッション番号（01/02）を媒介として Target/Control を決定する。

## 4. HEPイベント生成（R波→EEGサンプル）
### 4.1 サンプリング周波数と変換
仕様として以下を仮定する:
- ECG JSONの `peak_indices` は **500 Hz** サンプル
- EEG（前処理後の `*_clean.fif`）は **250 Hz**

変換は単純に
- `samples_250 = round(peak_indices_500hz / 2.0)`

とし、EEGの範囲外（`<0` または `>= raw.n_times`）は除外する。

※オフセット補正は行わない（セッション内で既にクロップ済み・同期済みである前提）。

### 4.2 MNE events 配列
- shape: `(n_events, 3)`
- `events[:, 0]`: R波サンプル（250Hz）
- `events[:, 2]`: `event_id`（既定 `999`）

## 5. エポック化と平均化（Evoked生成）
### 5.1 チャンネル
- `picks='eeg'`

### 5.2 エポック窓・ベースライン・除外
デフォルト（=標準仕様）:
- `tmin = -0.3` 秒
- `tmax = 0.8` 秒
- `baseline = (-0.2, -0.05)` 秒（0 msを含まない）
- `reject_eeg = 100e-6` Volt（100 µV）
- `event_id = 999`

実装:
- `mne.Epochs(..., reject_by_annotation=True, preload=True, detrend=None)`
- `epochs.average()` により `Evoked` を生成

### 5.3 全ドロップ時の挙動
もし `len(epochs) == 0`（全エポックが除外）となった場合は、
- drop-log 図を保存（`*_hep_drop_log.png/.svg`）
- 例外で停止し、原因（閾値が厳しい等）をガイドする

## 6. ROI（図用）
HEP生成モジュールは、統計用ROIとは別に「被験者内QC図」用のROI平均図を出す。

- `--roi` 未指定時のデフォルトROI: `["Fz", "Cz", "Pz", "C3", "C4"]`（存在するものだけ使用）
- もし1つも存在しない場合:
  - `Cz` があれば `Cz`
  - それも無ければ最初のEEGチャンネル

## 7. 出力
### 7.1 図（セッション単位）
保存先（バッチ時の既定）:
- `data/processed/<subject_id>/figures/`

出力（各sessごと）:
- Butterfly: `<stem>_hep.png`, `<stem>_hep.svg`
- ROI平均: `<stem>_hep_roi.png`, `<stem>_hep_roi.svg`

※`<stem>` は `*_clean` を除いたFIF stem（例: `<subject>_sess01`）。

### 7.2 Evoked（Target/Control集約）
保存先（バッチ時）:
- `data/processed/<subject_id>/<subject_id>_hep_target-ave.fif`
- `data/processed/<subject_id>/<subject_id>_hep_control-ave.fif`

保存関数:
- `mne.write_evokeds(path, [evoked])`

集約方法:
- Targetに分類されたセッションEvokedを重み付き平均（重みは `nave`、最低1）
- Controlも同様

再計算回避:
- 両方のEvokedが存在し、`--overwrite` が無い場合は被験者全体をスキップ

## 8. CLI仕様（主要）
- 単一処理: `--fif <...> --ecg_file <...>`
- バッチ処理: `--subject_id <id>`（推奨）

主なオプション:
- `--tmin`, `--tmax`, `--baseline`, `--reject_eeg`, `--event_id`
- `--roi ...`
- `--overwrite`
