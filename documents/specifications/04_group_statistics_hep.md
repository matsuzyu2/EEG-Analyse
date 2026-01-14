# 04_group_statistics_hep.md — 統計解析仕様（between-subject / HEP）

対象スクリプト: [src/05_group_statistics_hep.py](../../src/05_group_statistics_hep.py)

作成日: 2026-01-12

## 1. 概要（Overview）
本スクリプトは、HEP（Heartbeat Evoked Potential）について **Good Responders 群**と **Non-Good Responders 群**の間で、被験者間（between-subjects）の群間差があるかを検証する。

- 比較は条件ごとに独立に実施する。
  - Target条件: Good(Target) vs Non-Good(Target)
  - Control条件: Good(Control) vs Non-Good(Control)
- 統計手法は **独立2標本クラスター置換検定**（`mne.stats.permutation_cluster_test`）を用いる。
- 既に作成済みの中間生成物（HEP Evoked）を読み込むだけで、HEPの再計算は行わない。

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
被験者 `sid` について、以下の Evoked を使用する。

- Target: `data/processed/<sid>/<sid>_hep_target-ave.fif`
- Control: `data/processed/<sid>/<sid>_hep_control-ave.fif`

読み込み:
- `mne.read_evokeds(path, condition=0)`
- MNEのAPI差分により戻り値が `list[Evoked]` になる場合があるため、先頭要素を採用する。

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
- 解析段階で Good/Non-Good いずれかが2名未満の場合、例外で停止する。

### 3.2 ROI定義とROI平均（Data shaping）
ROIはスクリプト内辞書 `roi_dict` として定義する（固定）。

- `Frontal`: `["Fz", "F1", "F2", "F3", "F4", "FCz", "FC1", "FC2", "Cz"]`（候補）
- `Visual`: `["Oz", "O1", "O2"]`
- `Parietal`: `["Pz", "P3", "P4", "CPz", "CP1", "CP2"]`（候補）

備考:
- ROIの定義は「候補チャンネル」のリストとして持ち、被験者ごとのEvokedに存在しないチャンネルは自動的に除外して平均する（存在chのみで平均）。
- 候補が多いのは、計測モンタージュ差（例: `F1/F2/FCz/Pz` が無い）に対して解析を壊さないため。

ROI平均（Evoked → 1D時系列）:
- EvokedからROIチャンネルを `pick` し、`data.mean(axis=0)` を取る。
- ROIに含まれるチャンネルが **一つも存在しない**場合は例外で停止する。
- ROIチャンネルの一部が存在しない場合は、存在するチャンネルのみで平均を取る（完全一致は要求しない）。

得られるデータ形状:
- `X_good`: `(n_good, n_times)`
- `X_nongood`: `(n_nongood, n_times)`

### 3.3 時間軸統一（Interpolation / Alignment）
被験者間でEvokedの `times`（時間グリッド）が異なる可能性があるため、ROI時系列を参照グリッドへ統一する。

参照時間 `ref_times` の決定:
- 条件（Target/Control）ごとに、Good+Non-Good の候補被験者を先頭から走査し、該当Evokedが読めた最初の被験者の `ev.times` を参照グリッドとして採用する。
- 参照が一人も見つからない場合は例外で停止する。

線形補間（`np.interp`）:
- 被験者の `times` が `ref_times` と一致しない場合、線形補間で `ref_times` 上に再サンプリングする。
- 前提条件として、被験者の時間軸範囲が参照範囲を**完全に覆う必要**がある。
  - `subj_times[0] > ref_times[0]` または `subj_times[-1] < ref_times[-1]` の場合は、警告ログの上で当該被験者をスキップする。
- `times` が単調増加でない場合は、ソートしてから補間する。

補間後:
- 有効被験者が0になった場合は例外で停止する。

### 3.4 統計解析（Independent samples cluster permutation test）
条件ごとに独立に、Good群とNon-Good群を比較する。

- 関数: `mne.stats.permutation_cluster_test`
- 入力: `[X_good, X_nongood]`
- `adjacency=None`（隣接行列は使わない）
- `out_type="mask"`（クラスターをマスクとして受け取る指定）
- 検定統計量: `mne.stats.ttest_ind_no_p`（2標本t検定のt値、p値は返さない）

#### 3.4.1 比較パターン
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
- 有意クラスターのマスクをOR結合し、`sig_mask`（time方向1Dのbool配列）を作る。

#### 3.4.4 クラスター表現の正規化
MNEのバージョン差等により、`clusters` の返り値表現（mask / slice / index配列など）が揺れる可能性がある。
本スクリプトでは、内部でクラスターを **1Dのboolマスク**に正規化してから、
サマリ作成・NPZ保存・有意マスク生成を行う。

## 4. 出力（Outputs）

### 4.1 出力先ディレクトリ
- 既定: `data/group_analysis_between_subjects/`
- `--out_dir` で変更可能。

### 4.2 生成されるファイル
ROI（複数指定可）× 条件（Target/Control）ごとに以下を生成する。

#### 4.2.1 プロット画像（PNG）
- `hep_group_target_Good_vs_NonGood_<ROI>.png`
- `hep_group_control_Good_vs_NonGood_<ROI>.png`

図の内容:
- Good平均波形（赤）+ SEMシェーディング
- Non-Good平均波形（青）+ SEMシェーディング
- time=0ms の縦線
- 有意区間（`sig_mask`）を図の下部に太線バーとして表示（連続区間は自動的にまとめる）

注意:
- 波形は `µV` 表示のため、内部データ（V）を $\times 10^6$ して表示する。

#### 4.2.2 中間データ（NPZ, 再描画用）
ファイル名:
- `stats_<ROI>_target.npz`
- `stats_<ROI>_control.npz`

主な保存キー（代表）:
- メタ情報: `condition`, `roi`, `roi_channels`, `roi_candidates`, `times_s`
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
- `times_s`: `(n_times,)`
- `X_good`: `(n_good, n_times)`
- `X_nongood`: `(n_nongood, n_times)`
- `clusters`: `(n_clusters, n_times)`（クラスター数が0の場合は `(0, n_times)`）
- `sig_mask`: `(n_times,)`

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
  - `Time_Range`（ms、クラスターmaskの最初〜最後の時刻）

## 5. CLIオプション（Command Line Arguments）

### 5.1 主要引数とデフォルト
- `--good_csv`（default: `data/classification/good_responders_median.csv`）
  - Good Responders を定義するCSV。
- `--processed_dir`（default: `data/processed/`）
  - 被験者ごとの中間生成物ディレクトリ。
- `--out_dir`（default: `data/group_analysis_between_subjects/`）
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
  - `python src/05_group_statistics_hep.py --roi Frontal Visual Parietal`

- 出力先を指定:
  - `python src/05_group_statistics_hep.py --out_dir data/group_analysis_between_subjects_v2 --roi Visual`

- 特定被験者を除外:
  - `python src/05_group_statistics_hep.py --roi Visual --exclude 251210_MS 251215_FK`

## 6. 再描画手順（NPZから再解析なしでプロット生成）
本スクリプトは統計結果・描画に必要な情報を `stats_<ROI>_<condition>.npz` に保存する。
このNPZを読み込むことで、**統計の再実行なし**に平均波形・SEM・有意区間バーを再描画できる。

対象ファイル例:
- Target: `data/group_analysis_between_subjects/stats_Visual_target.npz`
- Control: `data/group_analysis_between_subjects/stats_Visual_control.npz`

### 6.1 NPZから再描画に使う主要キー
- `times_s`（秒）: x軸
- `good_mean`, `nongood_mean`（V）: 平均波形
- `good_sem`, `nongood_sem`（V）: SEM
- `sig_mask`（bool, time方向）: 有意区間バー
- `roi`, `condition`, `roi_channels`: タイトル/凡例用

### 6.2 最小の再描画例（Matplotlib）
以下は、NPZを読み込んで `src/05_group_statistics_hep.py` と同等の情報（平均+SEM+有意バー）を描く最小例である。

```python
import numpy as np
import matplotlib.pyplot as plt

npz_path = "data/group_analysis_between_subjects/stats_Visual_target.npz"
d = np.load(npz_path, allow_pickle=False)

times_ms = d["times_s"] * 1e3
good_mean_uv = d["good_mean"] * 1e6
nongood_mean_uv = d["nongood_mean"] * 1e6
good_sem_uv = d["good_sem"] * 1e6
nongood_sem_uv = d["nongood_sem"] * 1e6
sig = d["sig_mask"].astype(bool)

fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
ax.plot(times_ms, good_mean_uv, color="red", lw=2, label="Good")
ax.fill_between(times_ms, good_mean_uv - good_sem_uv, good_mean_uv + good_sem_uv, color="red", alpha=0.2, linewidth=0)
ax.plot(times_ms, nongood_mean_uv, color="blue", lw=2, label="Non-Good")
ax.fill_between(times_ms, nongood_mean_uv - nongood_sem_uv, nongood_mean_uv + nongood_sem_uv, color="blue", alpha=0.2, linewidth=0)
ax.axvline(0, color="k", lw=1, alpha=0.6)

# 有意区間バー（連続区間をまとめる）
if sig.any():
  ymin, ymax = ax.get_ylim()
  bar_y = ymin + 0.05 * (ymax - ymin)
  idx = np.where(sig)[0]
  starts = [idx[0]]
  ends = []
  for i in range(1, len(idx)):
    if idx[i] != idx[i - 1] + 1:
      ends.append(idx[i - 1])
      starts.append(idx[i])
  ends.append(idx[-1])
  for s, e in zip(starts, ends):
    ax.hlines(y=bar_y, xmin=times_ms[s], xmax=times_ms[e], colors="k", linewidth=6)

roi = str(d["roi"])
cond = str(d["condition"])
roi_chs = ", ".join(list(d["roi_channels"].astype(str)))
ax.set_title(f"HEP {cond} ROI={roi} ({roi_chs})")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Amplitude (µV)")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right")

fig.savefig("replot_stats_Visual_target.png", dpi=300, bbox_inches="tight")
plt.close(fig)
```

### 6.3 注意点
- 単位: NPZ内の波形は **V**、描画では **µV** に変換する（$\times 10^6$）。
- `sig_mask` は「`cluster_p_values < alpha` のクラスターをOR結合」した結果であり、クラスター単位の輪郭描画ではなく、時間方向バー表示に直接使うことを想定している。
- `clusters`（`(n_clusters, n_times)`）も保存されるため、必要であれば `cluster_p_values` と組み合わせて任意の可視化（例: 特定クラスターだけのバー）に拡張可能。
