# 01_tfr_analysis.md — 周波数解析仕様（TFR）

対象スクリプト: [src/02_analysis_tfr.py](../../src/02_analysis_tfr.py)

作成日: 2026-01-11

## 1. 目的
各被験者のセッションEEGに対し、Feedback提示（time=0）にロックした時間周波数解析（TFR）を行い、
セッション種別に基づく Target/Control 条件で **AverageTFR を中間保存**する。

本モジュールは「計算」担当であり、群レベル統計は行わない。

## 2. 入力
### 2.1 前処理済みEEG
- 既定: `data/processed/<subject_id>/` 配下の `*_clean.fif`
- 主に想定するファイル名:
  - `data/processed/<subject_id>/<subject_id>_sess01_clean.fif`
  - `data/processed/<subject_id>/<subject_id>_sess02_clean.fif`

### 2.2 条件マニフェスト（必須: subject_idモード）
- `data/processed/<subject_id>/conditions_manifest.json`
- 必須キー:
  - `session_condition_map`: `{"sess01": <Set1_Cond>, "sess02": <Set2_Cond>}`

Target/Control分類は `session_condition_map` の値（例: Increase/Decrease/Control）を以下で解釈する。
- Target: `increase`, `decrease`, `inc`, `dec`
- Control: `control`

## 3. 時間基準（time=0の定義と実装）
### 3.1 定義
TFRの time=0 は **Feedback提示時刻**である。

### 3.2 実装ロジック
本スクリプトは外部Trigger CSVを直接読むのではなく、前処理が注入した注釈を利用する。

- 前処理により、外部トリガーCSVの情報は `*_clean.fif` 内の `Annotations` として保存される
- その注釈のうち以下の **文字列description** をイベント化し、これを feedback onset として扱う
  - `TRIG/Feedback_High`
  - `TRIG/Feedback_Low`

イベント生成は `mne.events_from_annotations` を用い、`event_id` は descriptionごとに自動割当（1,2）される。

結果として、エポックは
- time=0: Feedback onset（High/Lowいずれも同じ“Feedback”として扱い、両方を含めて平均化）
- 解析範囲: `tmin`〜`tmax`

で切り出される。

## 4. 解析手法（Morlet Wavelet）
### 4.1 使用チャンネル
- `picks='eeg'`（EEG全チャンネル）

※ROI平均は統計側（04）で行う設計。TFR生成時点では全EEGを保持する。

### 4.2 TFR計算
Morlet変換は MNE の `mne.time_frequency.tfr_morlet` を用いる。

- 出力: `power`（ITCは計算しない）
- `return_itc=False`
- `output='power'`
- `average=True`（EpochsTFRではなくAverageTFRを生成）

### 4.3 パラメータ（仕様として固定）
CLIのデフォルト値（=標準解析条件）は以下。

- Epoch window
  - `tmin = -0.5` 秒
  - `tmax = 1.5` 秒
- Baseline
  - `baseline = None`（未適用）
  - 指定する場合: `--baseline <bmin> <bmax>`（例 `--baseline -0.3 0.0`）
- Rejection
  - `reject_eeg = None`（振幅閾値による除外なし）
  - 指定例: `--reject_eeg 150e-6`（Volt）
- Frequency grid
  - `fmin = 1.0` Hz
  - `fmax = 45.0` Hz
  - `fstep = 1.0` Hz
  - 実体: `freqs = np.arange(fmin, fmax + 1e-9, fstep)`
- Cycles
  - `n_cycles_div = 2.0`
  - 実体: `n_cycles = max(freqs / n_cycles_div, 1.0)`
    - 例: 10 Hz → 5 cycles
- Downsampling
  - `decim = 1`（間引きなし）
- Parallel
  - `n_jobs = 1`（`-1` で全コア）
- Convolution
  - `use_fft = False`（`--use_fft` を付けた場合のみ True）

## 5. 条件分割と保存戦略
### 5.1 条件分割（セッション単位）
入力はセッション単位FIF（`sess01`, `sess02`）であり、各セッションの条件（Target/Control）は
`conditions_manifest.json` の `session_condition_map` から決定する。

### 5.2 平均化（被験者内でTarget/Controlに集約）
- Targetに該当するセッションTFR群を重み付き平均
- Controlに該当するセッションTFR群を重み付き平均

重みは `AverageTFR.nave` を用いる（最低1にクリップ）。

### 5.3 保存形式（AverageTFR .h5）
- 保存関数: `mne.time_frequency.write_tfrs(path, tfr)`
- データ型: `float32` にキャストしてディスクサイズを抑制

保存先（subject_idモード）:
- `data/processed/<subject_id>/<subject_id>_tfr_target.h5`
- `data/processed/<subject_id>/<subject_id>_tfr_control.h5`

TFRには provenance として `comment` に
`"<subject> <sessXX> <Target|Control> (Feedback-locked)"`
を設定する（可能な場合）。

### 5.4 再計算回避
- `*_tfr_target.h5` と `*_tfr_control.h5` が両方存在し、かつ `--overwrite` がない場合は被験者全体をスキップ

## 6. CLI仕様（主要）
- 被験者単位: `--subject_id <id>`（推奨）
- 単一FIF: `--fif <path>`
  - manifest が無い場合は `--session_type Increase|Decrease|Control` が必要

主なオプション:
- `--overwrite`
- `--tmin`, `--tmax`, `--baseline`
- `--fmin`, `--fmax`, `--fstep`, `--n_cycles_div`
- `--decim`, `--n_jobs`, `--use_fft`

## 7. 失敗条件とエラーハンドリング
- `conditions_manifest.json` が無い（subject_idモード）
- `*_clean.fif` が見つからない
- `TRIG/Feedback_High/Low` が注釈として存在せず、イベントが作れない
- `session_condition_map` の値が Increase/Decrease/Control として解釈できない

上記は例外として停止する（統計側の混乱を避ける設計）。
