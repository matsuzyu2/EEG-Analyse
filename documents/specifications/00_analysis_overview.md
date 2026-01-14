# 00_analysis_overview.md — システム概要

作成日: 2026-01-11

## 1. 目的（研究上の問い）
本パイプラインの目的は、**Good Responders** における「心拍制御（Target） vs Control」条件の差分から、

- 脳活動（EEG）の時間周波数特性（TFR: Time-Frequency Representation）
- 心拍誘発電位（HEP: Heartbeat-Evoked Potential）

を **被験者内（within-subject）** で比較し、条件差に一貫した時間・周波数・時系列クラスターが存在するかを検証・可視化することです。

本解析では、**Target/Control の定義は「Feedback_High/Low」ではなくセッション種別**により行います。
- Target: `Increase` / `Decrease`（大文字小文字や略記 `inc/dec` も許容）
- Control: `Control`

## 2. アーキテクチャ（全体データフロー）
本システムは「重い計算は中間保存し、統計は読み込みのみ」で完結する疎結合構成です。

### 2.1 全体フロー
1. **前処理（Preprocessing）**: セッションごとの `*_clean.fif` とイベント（注釈）を生成
2. **中間解析・保存（個別被験者）**
   - TFR解析: `AverageTFR` を Target/Control で分割保存
   - HEP解析: `Evoked` を Target/Control で分割保存
3. **統計解析（群レベル / within-subject）**
   - Good Responders のみ抽出
   - Target−Control 差分に対する 1標本クラスター置換検定
   - 図と最小統計（mask/p値）を出力

### 2.2 モジュールと責務
- `src/01_preprocess.py`
  - 外部トリガーCSVをEEG時間系へ写像し、`TRIG/` 系アノテーションを注入
  - セッション切り出し、リサンプリング、フィルタ、ASR/ICA等の前処理を実施
  - 出力: `data/processed/<subject_id>/<subject_id>_sessXX_clean.fif` など

- `src/02_analysis_tfr.py`（TFR中間生成）
  - `TRIG/Feedback_High` / `TRIG/Feedback_Low` を time=0 としてエポック化
  - MorletによるTFRを計算し、Target/Control で `AverageTFR` を保存

- `src/03_analysis_hep.py`（HEP中間生成）
  - ECGピーク（R波）からエポック化して `Evoked` を生成
  - セッション種別により Target/Control を集約し `Evoked` を保存

- `src/04_condition_statistics.py`（統計・可視化）
  - Good Responders を抽出し、中間ファイルのみ読み込み
  - ROI平均後にクラスター置換検定 → 図・統計を出力

- `src/00_batch_run.py`（オーケストレーション）
  - `data/conditions/conditions.csv` を走査し、上記を順次実行
  - `data/processed/<id>/conditions_manifest.json` を生成（条件マッピングの唯一の参照元）

## 3. データ配置と命名規約（I/Oの要点）

### 3.1 入力
- 前処理済みEEG: `data/processed/<subject_id>/<subject_id>_sess01_clean.fif`, `..._sess02_clean.fif`
- 条件マニフェスト: `data/processed/<subject_id>/conditions_manifest.json`
- ECGピーク: `data/ecg/<subject_id>/*Session_01*.json`, `*Session_02*.json`
- Good Responders: `data/classification/good_responders_median.csv`（`session_id` 列を subject_id として扱う）

### 3.2 中間生成物
- TFR（AverageTFR, HDF5）
  - `data/processed/<subject_id>/<subject_id>_tfr_target.h5`
  - `data/processed/<subject_id>/<subject_id>_tfr_control.h5`
- HEP（Evoked, FIF）
  - `data/processed/<subject_id>/<subject_id>_hep_target-ave.fif`
  - `data/processed/<subject_id>/<subject_id>_hep_control-ave.fif`

### 3.3 統計出力
- 出力先（デフォルト）: `data/group_analysis_within_subject/`
- 図（ROIごと）
  - TFR: `tfr_<ROI>_target_control_diff.png`
  - HEP: `hep_<ROI>_target_control.png`
- 最小統計
  - `stats_<ROI>.npz`（cluster p-values と有意mask）

## 4. 前処理仕様の参照（重要）
前処理（Triggerアラインメント、セッション切り出し、ASR/ICA、`TRIG/Feedback_*` 注釈注入）については、既存ドキュメントを参照してください。本書では詳細を重複記述しません。

- 前処理ドキュメント: [documents/EEG_Preprocessing.md](../../src/documents/EEG_Preprocessing.md)
- 参考ノートブック: [src/EEG_preprocessing.ipynb](../../src/EEG_preprocessing.ipynb)

## 5. 実行順（最小）
1. バッチ（推奨）: `src/00_batch_run.py`
2. もしくは個別に
   - `src/02_analysis_tfr.py`（TFR生成）
   - `src/03_analysis_hep.py`（HEP生成）
   - `src/04_condition_statistics.py`（統計）

本システムは 02/03 の保存物を前提に 04 が動作する設計です（04は再計算を行いません）。
