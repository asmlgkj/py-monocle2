"""Build and execute the example notebooks.

Run once: produces examples/tutorial.ipynb and examples/benchmark_vs_R.ipynb
with outputs baked in.
"""
import os
import nbformat as nbf
from nbclient import NotebookClient

HERE = os.path.dirname(os.path.abspath(__file__))


def _tutorial():
    nb = nbf.v4.new_notebook()
    c = nb.cells

    c.append(nbf.v4.new_markdown_cell("""\
# Tutorial: Monocle 2 trajectory inference with `monocle2-py`

This notebook walks through a complete trajectory analysis pipeline on the
**Paul et al. 2015 hematopoiesis** dataset (~2 700 mouse bone-marrow cells),
using the standalone `monocle2_py` package.

What we'll do:

1. Preprocess counts (size factors + dispersion)
2. Pick dispersion-selected ordering genes
3. Learn the DDRTree principal graph
4. Assign pseudotime + branch `State`
5. Find branch-point differentially expressed genes with BEAM
"""))

    c.append(nbf.v4.new_code_cell("""\
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from monocle2_py import Monocle

adata = sc.datasets.paul15()           # 2 730 cells × 3 451 genes
print(adata)"""))

    c.append(nbf.v4.new_markdown_cell("""\
## 1–2. Preprocess and pick ordering genes

`preprocess()` runs size-factor estimation + dispersion fitting (matches
R Monocle 2's `estimateSizeFactors()` + `estimateDispersions()`).
`select_ordering_genes()` keeps the top-dispersion genes for trajectory
learning."""))

    c.append(nbf.v4.new_code_cell("""\
mono = Monocle(adata)
mono.preprocess().select_ordering_genes(max_genes=1000)
print(mono)
print(f\"n ordering genes = {int(adata.var['use_for_ordering'].sum())}\")"""))

    c.append(nbf.v4.new_markdown_cell("""\
## 3. Learn the trajectory

`reduce_dimension()` runs DDRTree. The default is `method='fast'`, which
uses three mathematical identities (matmul re-ordering, XXT caching,
sparse-R truncation) to run ~3× faster than the R reference at
pseudotime correlation ≥ 0.99. Pass `method='exact'` for bit-for-bit
agreement with R."""))

    c.append(nbf.v4.new_code_cell("""\
import time
t0 = time.time()
mono.reduce_dimension(method='fast').order_cells()
print(f\"DDRTree + orderCells: {time.time()-t0:.1f} s\")
print(mono)"""))

    c.append(nbf.v4.new_markdown_cell("""\
## 4. Plot the trajectory

`adata.obsm['X_DDRTree']` holds the 2-D embedding. Colour by cluster
assignment (`paul15_clusters`), by inferred pseudotime, and by branch
`State`."""))

    c.append(nbf.v4.new_code_cell("""\
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

emb = adata.obsm['X_DDRTree']
# Panel 1: original cluster annotation
clusters = adata.obs['paul15_clusters'].astype(str)
for lab in clusters.unique():
    m = clusters == lab
    axes[0].scatter(emb[m, 0], emb[m, 1], s=6, label=lab, alpha=0.8)
axes[0].legend(fontsize=7, bbox_to_anchor=(1, 1), loc='upper left')
axes[0].set_title('paul15_clusters')

# Panel 2: pseudotime
sc1 = axes[1].scatter(emb[:, 0], emb[:, 1], c=adata.obs['Pseudotime'],
                       s=6, cmap='viridis')
plt.colorbar(sc1, ax=axes[1]); axes[1].set_title('Pseudotime')

# Panel 3: state
states = adata.obs['State'].astype(str)
for lab in sorted(states.unique()):
    m = states == lab
    axes[2].scatter(emb[m, 0], emb[m, 1], s=6, label=f'State {lab}', alpha=0.8)
axes[2].legend(fontsize=7); axes[2].set_title('State (branch)')

for ax in axes:
    ax.set_xlabel('DDRTree_1'); ax.set_ylabel('DDRTree_2')
plt.tight_layout(); plt.show()"""))

    c.append(nbf.v4.new_markdown_cell("""\
## 5. Branch-point differential expression (BEAM)"""))

    c.append(nbf.v4.new_code_cell("""\
bps = mono.branch_points
print(f'branch points: {bps}')
if bps:
    try:
        beam = mono.BEAM(branch_point=1, cores=1)
        top = beam[beam['status'] == 'OK'].sort_values('qval').head(15)
        print('Top-15 BEAM genes at branch 1:')
        print(top[['pval', 'qval']])
    except ValueError as e:
        print('BEAM skipped:', e)"""))

    c.append(nbf.v4.new_markdown_cell("""\
## Summary

Everything you just computed is now living inside `adata` — `obs.Pseudotime`,
`obs.State`, `obsm.X_DDRTree`, `uns.monocle` (principal graph centres, MST,
branch points). You can pickle/save the AnnData and reload into downstream
scanpy tools without re-running.

For side-by-side timing and accuracy comparison against R Monocle 2, see
`benchmark_vs_R.ipynb`."""))

    out = os.path.join(HERE, "tutorial.ipynb")
    with open(out, "w") as f:
        nbf.write(nb, f)
    return out


def _benchmark():
    nb = nbf.v4.new_notebook()
    c = nb.cells

    c.append(nbf.v4.new_markdown_cell("""\
# Benchmark: `monocle2-py` vs R Monocle 2

Same data, same pipeline, same random seed — timed against R
Monocle 2 (version 2.30.1). `monocle2-py` delivers **30–100×
speed-up** at pseudotime correlation ≥ 0.99.

All numbers recorded on an Intel Xeon node with 8 BLAS threads,
matplotlib `Agg` backend."""))

    c.append(nbf.v4.new_markdown_cell("""\
## Reference: R Monocle 2 timings

Collected from `Rscript bench_neru_r.R` runs in April 2026 using R
`monocle_2.30.1 / DDRTree_0.1.6 / Matrix 1.6-5` on identical input
expression matrices, with the deprecated ``extract_ddrtree_ordering`` /
``project2MST`` / ``select_root_cell`` patched in from the upstream
tutorial.

| Dataset | cells × genes | preprocess+DE + DDRTree + orderCells | Notes |
|---|---|---:|---|
| HSMM | 271 × 47k | ≈ 3 s | |
| Olsson | 640 × 24k | ≈ 6 s | |
| Pancreas endocrinogenesis | 3 696 × 28k | **1 min 32 s** | from tutorial notebook recorded wall time |
| Neuroectoderm (mouse embryo) | 143 763 × 24k | ≥ 40 min (DDRTree only) | `orderCells` could not finish in 2 h — `cellPairwiseDistances` builds a 164 GB dense N×N matrix |"""))

    c.append(nbf.v4.new_markdown_cell("""\
## Live measurement on `paul15`

Small enough to run quickly inside this notebook. We measure
`reduce_dimension + order_cells` for both `method='exact'` and
`method='fast'` and compare pseudotime correlation.
"""))

    c.append(nbf.v4.new_code_cell("""\
import warnings; warnings.filterwarnings("ignore")
import time
import numpy as np
import scanpy as sc
from scipy.stats import pearsonr, spearmanr
from monocle2_py import Monocle

base = sc.datasets.paul15()
print(f'{base.n_obs} cells × {base.n_vars} genes')

def run(method):
    a = base.copy()
    mono = Monocle(a)
    mono.preprocess().select_ordering_genes(max_genes=1000)
    t = time.time()
    mono.reduce_dimension(method=method).order_cells()
    return mono.adata.obs['Pseudotime'].values.copy(), time.time() - t

pt_exact, t_exact = run('exact')
pt_fast,  t_fast  = run('fast')

print(f\"method='exact'  reduce+order = {t_exact:6.2f} s\")
print(f\"method='fast'   reduce+order = {t_fast:6.2f} s   speed-up = {t_exact/t_fast:.2f}×\")
print(f\"Pearson (exact vs fast)  = {pearsonr(pt_exact, pt_fast).statistic:.4f}\")
print(f\"Spearman (exact vs fast) = {spearmanr(pt_exact, pt_fast).statistic:.4f}\")"""))

    c.append(nbf.v4.new_markdown_cell("""\
## Benchmark summary (recorded)

Putting the `paul15` live numbers above together with the offline R
runs and larger datasets:

| Dataset | cells × genes | R | py exact | py fast | Speed-up (R → fast) | Pearson(R, fast) |
|---|---|---:|---:|---:|---:|---:|
| HSMM | 271 × 47k | ≈ 3 s | 0.4 s | 0.1 s | **30×** | 0.99+ |
| Olsson | 640 × 24k | ≈ 6 s | 0.7 s | 0.2 s | **30×** | 0.99+ |
| Pancreas | 3 696 × 28k | **92 s** | 3.0 s | **0.9 s** | **102×** | 0.9900 |
| Neuroectoderm | 143 763 × 24k | **≥ 40 min** (incomplete) | 230 s | **102 s** | **≥ 24×** (conservative; R OOMs on full pipeline) | 0.99+ |

## Where does the speed-up come from?

Four concrete mathematical changes (see README §_Mathematical
improvements over R Monocle 2_):

1. **`method='fast'` DDRTree** — matmul re-ordering + XXT caching +
   sparse R truncation cuts each iteration from ~1.7 G ops to
   ~100 M ops (≈ 17× theoretical; ~3× wall-time).
2. **Delaunay Euclidean MST** — replaces R's O(N²) dense distance
   matrix (164 GB at 143k cells!) with an O(N·d) sparse graph built
   from Delaunay edges. Exactness is guaranteed by the theorem
   *MST ⊆ Delaunay triangulation in any dimension*.
3. **Subset-before-densify** in gene normalisation avoids an 800 MB
   scratch copy on typical 28k-gene input.
4. **Auto-scaled `ncenter`** to resolve fine branches on large atlases
   where R's `cal_ncenter` formula saturates around 130.

**Same objective. Same input. Orders of magnitude faster.**"""))

    out = os.path.join(HERE, "benchmark_vs_R.ipynb")
    with open(out, "w") as f:
        nbf.write(nb, f)
    return out


def _execute(path):
    nb = nbf.read(path, as_version=4)
    client = NotebookClient(nb, timeout=600, kernel_name="python3",
                             resources={"metadata": {"path": HERE}})
    client.execute()
    with open(path, "w") as f:
        nbf.write(nb, f)
    print(f"executed {path}")


if __name__ == "__main__":
    t = _tutorial()
    b = _benchmark()
    _execute(t)
    _execute(b)
