# 05_group_statistics_tfr.md — 統計解析仕様（between-subject / TFR）

対象スクリプト: [src/06_group_statistics_tfr.py](../../src/06_group_statistics_tfr.py)

作成日: 2026-01-14

## 1. 概要（Overview）
本スクリプトは、TFR（Time-Frequency Representation）について **Good Responders 群**と **Non-Good Responders 群**の間で、被験者間（between-subjects）の群間差があるかを検証する。

- 比較は条件ごとに独立に実施する。
  - Target条件: Good(Target) vs Non-Good(Target)
  - Control条件: Good(Control) vs Non-Good(Control)
- 統計手法は **独立2標本クラスター置換検定**（`mne.stats.permutation_cluster_test`）を用いる。
- 既に作成済みの中間生成物（TFR H5ファイル）を読み込むだけで、TFRの再計算は行わない。
- **HEP処理は一切含まない**（TFR専用スクリプト）。

## 2. 入力（Input）

### 2.1 Good Responders の定義元
- 既定: `data/classification/good_responders_median.csv`
- 必須列: `session_id`
- `session_id` は `subject_id` として扱う（`data/processed/<sid>/` のディレクトリ名と一致する前提）。
- CSV内のIDは、**順序保持・重複除去**して読み込む。

備考:
- CSVに載っていても `processed_dir` にディレクトリが存在しないIDは、警告ログを出してスキップする。

### 2.2 Non-Good Responders の定義ロジック
- `processed_dir`（既定: `data/processed/`）直下のサブディレクトリを全列挙し、そのうち **Goodに含まれない被験者ID**を Non-Good と定義する。
- `processed_dir` 直下のサブディレクトリは名前順にソートして扱う。

### 2.3 除外リスト（`--exclude`）
- `--exclude` で与えられた被験者IDは Good/Non-Good 両群から除外する。
- `--exclude` に指定されたIDがどちらの群にも存在しない場合は、警告ログを出して無視する（エラーにはしない）。

### 2.4 使用する中間ファイル（被験者ごと）
被験者 `sid` について、以下の TFR H5 ファイルを使用する。

- Target: `data/processed/<sid>/<sid>_tfr_target.h5`
- Control: `data/processed/<sid>/<sid>_tfr_control.h5`

読み込み:
- `mne.time_frequency.read_tfrs(path)`
- MNEのAPI差分により戻り値が `list[AverageTFR]` になる場合があるため、先頭要素を採用する。
- 戻り値が空リストの場合、またはAverageTFR型でない場合はエラーで停止する。

欠損:
- 対象ファイルが存在しない被験者は警告ログを出してスキップする。

## 3. 処理フロー（Processing Logic）

### 3.1 被験者選定（Good/Non-Good の振り分け）
1. `good_responders_median.csv` から Good候補のIDリストを読み取り（順序保持・重複除去）。
2. `processed_dir` 配下の被験者ディレクトリを列挙。
3. Good候補のうち、`processed_dir` に存在するものだけを Good として採用（存在しないものは警告して除外）。
4. Non-Good は「`processed_dir` に存在し、Good に含まれないもの」として定義。
5. `--exclude` を Good/Non-Good の両方に適用。

制約（統計が成立するための下限）:
- 解析段階で Good/Non-Good いずれかが2名未満の場合、例外で停止する（`RuntimeError`）。

### 3.2 ROI定義とROI平均（Data shaping）
ROIはスクリプト内辞書 `roi_dict` として定義する（固定）。

- `Frontal`: `["Fz", "F1", "F2", "F3", "F4", "FCz", "FC1", "FC2", "Cz"]`（候補）
- `Visual`: `["Oz", "O1", "O2"]`
- `Parietal`: `["Pz", "P3", "P4", "CPz", "CP1", "CP2"]`（候補）

備考:
- ROIの定義は「候補チャンネル」のリストとして持ち、被験者ごとのAverageTFRに存在しないチャンネルは自動的に除外して平均する（存在chのみで平均）。
- 候補が多いのは、計測モンタージュ差（例: `F1/F2/FCz/Pz` が無い）に対して解析を壊さないため。

ROI平均（AverageTFR → 2D時系列: freq × time）:
- AverageTFRからROIチャンネルを `pick` し、`data.mean(axis=0)` を取る（channel次元を平均）。
- ROIに含まれるチャンネルが **一つも存在しない**場合は例外で停止する。
- ROIチャンネルの一部が存在しない場合は、存在するチャンネルのみで平均を取る（完全一致は要求しない）。

得られるデータ形状:
- `X_good`: `(n_good, n_freqs, n_times)`
- `X_nongood`: `(n_nongood, n_freqs, n_times)`

### 3.3 時間・周波数グリッド統一（2D Interpolation / Alignment）
被験者間でAverageTFRの `freqs`（周波数グリッド）と `times`（時間グリッド）が異なる可能性があるため、ROI時系列を参照グリッドへ統一する。

#### 3.3.1 参照グリッド `(ref_freqs, ref_times)` の決定
- 条件（Target/Control）ごとに、Good+Non-Good の候補被験者を先頭から走査し、該当TFRが読めた最初の被験者の `(tfr.freqs, tfr.times)` を参照グリッドとして採用する。
- 参照が一人も見つからない場合は例外で停止する（`RuntimeError`）。

#### 3.3.2 2次元線形補間（`_align_tfr_roi_to_reference`）
被験者の `(freqs, times)` が `(ref_freqs, ref_times)` と一致しない場合、2次元線形補間で統一する。

前提条件:
- 被験者の周波数軸範囲が参照範囲を**完全に覆う必要**がある。
  - `subj_freqs[0] > ref_freqs[0]` または `subj_freqs[-1] < ref_freqs[-1]` の場合は、警告ログの上で当該被験者をスキップする。
- 被験者の時間軸範囲が参照範囲を**完全に覆う必要**がある。
  - `subj_times[0] > ref_times[0]` または `subj_times[-1] < ref_times[-1]` の場合は、警告ログの上で当該被験者をスキップする。

軸のソート:
- `freqs` が単調増加でない場合は、ソートしてから補間する（`np.argsort`）。
- `times` が単調増加でない場合は、ソートしてから補間する。

補間プロセス:
1. **周波数方向の補間**: 各時刻について、`freqs` → `ref_freqs` へ線形補間（`np.interp`）。
2. **時間方向の補間**: 各周波数について、`times` → `ref_times` へ線形補間。

補間後:
- 有効被験者が0になった場合は例外で停止する（`RuntimeError`）。

### 3.4 統計解析（Independent samples cluster permutation test）
条件ごと・ROIごとに独立に、Good群とNon-Good群を比較する。

- 関数: `mne.stats.permutation_cluster_test`
- 入力: `[X_good, X_nongood]`
- `adjacency=None`（隣接行列は使わない）
- `out_type="mask"`（クラスターをマスクとして受け取る指定）
- 検定統計量: `mne.stats.ttest_ind_no_p`（2標本t検定のt値、p値は返さない）

#### 3.4.1 比較パターン
各ROI・条件について:
- Target: `X_good_target` vs `X_nongood_target`
- Control: `X_good_control` vs `X_nongood_control`

#### 3.4.2 クラスター形成閾値（t分布）
クラスター形成は、t分布の臨界値に基づく。

- `n1 = n_good`, `n2 = n_nongood`
- 自由度: `df = n1 + n2 - 2`
- `cluster_alpha` と `tail` から閾値を決める。
  - `tail=0`（両側）: $t_{thr} = t_{ppf}(1 - cluster\_alpha/2, df)$
  - `tail=1`（右側）: $t_{thr} = t_{ppf}(1 - cluster\_alpha, df)$
  - `tail=-1`（左側）: $t_{thr} = t_{ppf}(cluster\_alpha, df)$

#### 3.4.3 有意マスク（`sig_mask`）
- クラスターごとのp値 `cluster_p_values` に対し、`p < alpha` のクラスターを有意とみなす。
- 有意クラスターのマスクをOR結合し、`sig_mask`（2D: freq × time のbool配列）を作る。

#### 3.4.4 クラスター表現の正規化（2D版）
MNEのバージョン差等により、`clusters` の返り値表現が揺れる可能性がある。
本スクリプトでは、内部でクラスターを **2D (freq × time) のboolマスク**に正規化してから、
サマリ作成・NPZ保存・有意マスク生成を行う。

サポートする形式:
- 2D bool array: そのまま使用
- 1D bool array (flattened): `reshape(n_freqs, n_times)` で2Dに変換
- Index array: フラット化インデックスからboolマスクを構築
- Tuple of (freq_indices, time_indices): 各要素ペアでマスク構築

## 4. 出力（Outputs）

### 4.1 出力先ディレクトリ
- 既定: `data/group_analysis_between_subjects_tfr/`
- `--out_dir` で変更可能。

### 4.2 生成されるファイル
ROI（複数指定可）× 条件（Target/Control）ごとに以下を生成する。

#### 4.2.1 プロット画像（PNG）
ファイル名:
- `tfr_<ROI>_target_Good_vs_NonGood.png`
- `tfr_<ROI>_control_Good_vs_NonGood.png`

図の内容（3面ヒートマップ）:
1. **左パネル**: Good平均（viridis カラーマップ）
2. **中央パネル**: Non-Good平均（viridis カラーマップ）
3. **右パネル**: 差分（Good - NonGood）（RdBu_r カラーマップ、対称スケール）

各パネルの仕様:
- x軸: Time (s)
- y軸: Frequency (Hz)
- time=0 の縦線（黒、半透明）
- 差分パネルのカラーバー範囲: `abs(diff_mean)` の98パーセンタイルで対称スケール（vmin=-vmax, vmax=vmax）
  - 98パーセンタイルが0以下の場合は、最大絶対値にフォールバック
- **有意クラスター**: 差分パネルに黒い輪郭線（`contour`）で重畳表示

#### 4.2.2 中間データ（NPZ, 再描画用）
ファイル名:
- `stats_<ROI>_target.npz`
- `stats_<ROI>_control.npz`

主な保存キー（代表）:
- メタ情報: `condition`, `roi`, `roi_channels`, `roi_candidates`, `freqs_hz`, `times_s`
- 被験者ID: `good_subjects`, `nongood_subjects`
- データ行列: `X_good`, `X_nongood`
- 記述統計: `good_mean`, `nongood_mean`, `good_sem`, `nongood_sem`
- 検定出力: `T_obs`, `cluster_p_values`, `clusters`, `sig_mask`
- 解析パラメータ: `n_permutations`, `cluster_alpha`, `alpha`, `tail`, `seed`

ROIチャンネル関連:
- `roi_candidates`: ROIの候補チャンネル（固定）
- `roi_channels`: 当該条件×ROIで、実データに存在し「実際に使われたチャンネル」（候補のサブセット）
- `good_roi_pick_mask`: `(n_good, n_candidates)` のbool。被験者ごとに、候補のうち採用されたチャンネル
- `nongood_roi_pick_mask`: `(n_nongood, n_candidates)` のbool。同上

形状の目安:
- `freqs_hz`: `(n_freqs,)`
- `times_s`: `(n_times,)`
- `X_good`: `(n_good, n_freqs, n_times)`
- `X_nongood`: `(n_nongood, n_freqs, n_times)`
- `good_mean`, `nongood_mean`: `(n_freqs, n_times)`
- `good_sem`, `nongood_sem`: `(n_freqs, n_times)`
- `T_obs`: `(n_freqs, n_times)`
- `clusters`: `(n_clusters, n_freqs, n_times)`（クラスター数が0の場合は `(0, n_freqs, n_times)`）
- `sig_mask`: `(n_freqs, n_times)` のbool

#### 4.2.3 サマリ（CSV）
- `statistics_summary.csv`

内容:
- 有意クラスター（`p_value < alpha`）のみを行として出力する。
- 列:
  - `ROI`
  - `Condition`（`Target` / `Control`）
  - `n_good`（実際に解析に使われた被験者数）
  - `n_nongood`（実際に解析に使われた被験者数）
  - `Cluster_ID`（有意クラスターの通し番号、条件×ROIごとに1から）
  - `p_value`
  - `Time_Range`（ms、クラスターmaskの時間方向の最小〜最大）
  - `Freq_Range`（Hz、クラスターmaskの周波数方向の最小〜最大）

## 5. CLIオプション（Command Line Arguments）

### 5.1 主要引数とデフォルト
- `--good_csv`（default: `data/classification/good_responders_median.csv`）
  - Good Responders を定義するCSV。
- `--processed_dir`（default: `data/processed/`）
  - 被験者ごとの中間生成物ディレクトリ。
- `--out_dir`（default: `data/group_analysis_between_subjects_tfr/`）
  - 出力先ディレクトリ。
- `--roi`（default: `Frontal`）
  - ROI名を1つ以上指定（例: `--roi Frontal Visual Parietal`）。
- `--exclude`（default: 空）
  - 除外する被験者ID（スペース区切り、例: `--exclude 251210_MS 251215_FK`）。

統計パラメータ:
- `--n_permutations`（default: `2000`）
- `--cluster_alpha`（default: `0.05`）
  - クラスター形成のt閾値を決めるためのalpha。
- `--alpha`（default: `0.05`）
  - クラスターp値の有意判定閾値。
- `--tail`（default: `0`）
  - `-1`（左側）, `0`（両側）, `1`（右側）。
- `--seed`（default: `0`）

### 5.2 実行例
- ROIを3つまとめて解析:
  - `python src/06_group_statistics_tfr.py --roi Frontal Visual Parietal`

- 出力先を指定:
  - `python src/06_group_statistics_tfr.py --out_dir data/group_analysis_between_subjects_tfr_v2 --roi Visual`

- 特定被験者を除外:
  - `python src/06_group_statistics_tfr.py --roi Visual --exclude 251210_MS 251215_FK`

## 6. 再描画手順（NPZから再解析なしでプロット生成）
本スクリプトは統計結果・描画に必要な情報を `stats_<ROI>_<condition>.npz` に保存する。
このNPZを読み込むことで、**統計の再実行なし**に3面ヒートマップを再描画できる。

対象ファイル例:
- Target: `data/group_analysis_between_subjects_tfr/stats_Visual_target.npz`
- Control: `data/group_analysis_between_subjects_tfr/stats_Visual_control.npz`

### 6.1 NPZから再描画に使う主要キー
- `freqs_hz`（Hz）: y軸
- `times_s`（秒）: x軸
- `good_mean`, `nongood_mean`（freq × time）: 各群の平均パワー
- `sig_mask`（bool, freq × time）: 有意クラスター領域
- `roi`, `condition`, `roi_channels`: タイトル用

### 6.2 最小の再描画例（Matplotlib）
以下は、NPZを読み込んで `src/06_group_statistics_tfr.py` と同等の3面ヒートマップを描く最小例である。

```python
import numpy as np
import matplotlib.pyplot as plt

npz_path = "data/group_analysis_between_subjects_tfr/stats_Visual_target.npz"
d = np.load(npz_path, allow_pickle=False)

freqs = d["freqs_hz"]
times = d["times_s"]
good_mean = d["good_mean"]
nongood_mean = d["nongood_mean"]
diff_mean = good_mean - nongood_mean
sig = d["sig_mask"].astype(bool)

vmax_diff = float(np.nanpercentile(np.abs(diff_mean), 98))
if vmax_diff <= 0:
    vmax_diff = float(np.nanmax(np.abs(diff_mean)) or 1.0)

fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

def _imshow(ax, data, title, cmap, vmin=None, vmax=None):
    im = ax.imshow(
        data,
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.axvline(0, color="k", lw=1, alpha=0.6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    return im

im0 = _imshow(axes[0], good_mean, "Good", cmap="viridis")
im1 = _imshow(axes[1], nongood_mean, "Non-Good", cmap="viridis")
im2 = _imshow(axes[2], diff_mean, "Good - NonGood", cmap="RdBu_r", vmin=-vmax_diff, vmax=vmax_diff)

# 有意クラスター輪郭線
if sig.any():
    axes[2].contour(times, freqs, sig.astype(float), levels=[0.5], colors=["k"], linewidths=1.5)

fig.colorbar(im0, ax=axes[0], fraction=0.046).set_label("Power")
fig.colorbar(im1, ax=axes[1], fraction=0.046).set_label("Power")
fig.colorbar(im2, ax=axes[2], fraction=0.046).set_label("Δ Power")

roi = str(d["roi"])
cond = str(d["condition"])
roi_chs = ", ".join(list(d["roi_channels"].astype(str)))
fig.suptitle(f"TFR ROI={roi} ({roi_chs})", y=1.02)

fig.savefig("replot_stats_Visual_target.png", dpi=300, bbox_inches="tight")
plt.close(fig)
```

### 6.3 注意点
- `sig_mask` は「`cluster_p_values < alpha` のクラスターをOR結合」した結果であり、クラスター単位の輪郭描画に直接使うことを想定している。
- `clusters`（`(n_clusters, n_freqs, n_times)`）も保存されるため、必要であれば `cluster_p_values` と組み合わせて任意の可視化（例: 特定クラスターだけの輪郭）に拡張可能。
- カラーマップ範囲は98パーセンタイルを使った対称スケールを推奨する（外れ値の影響を抑えるため）。
