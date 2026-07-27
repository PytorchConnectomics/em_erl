# Examples

These examples run directly from a source checkout because each script adds the
repository root to `sys.path`. They also work after `pip install -e .`.

## J0126 CloudVolume workflow (`eval_j0126.py`)

References: [FFN paper](https://www.nature.com/articles/s41592-018-0049-4), [J0126 data README](https://storage.googleapis.com/j0126-nature-methods-data/GgwKmcKgrcoNxJccKuGIzRnQqfit9hnfK1ctZzNbnuU/README.txt)

### Reproduced number

The FFN paper ([Januszewski et al., Nature Methods 2018](https://www.nature.com/articles/s41592-018-0049-4))
reports a **mean error-free neurite path length of 1.1 mm** on zebra finch, with only
4 mergers in **97 mm** of test paths.

Scoring the public FFN segmentation against the 50 test skeletons, with
`merge_threshold=0` (the paper's convention — every merge counts):

| resolution (x,y,z nm) | GT total path | gt ERL | ERL | NERL |
|---|---|---|---|---|
| `10 x 10 x 20` (paper) | **97.4 mm** | 2.114 mm | **1.105 mm** | 0.5228 |
| `9 x 9 x 20` | 91.1 mm | 1.978 mm | 1.033 mm | 0.5223 |

FFN hits the reported 1.1 mm exactly. Getting there needs two settings right:

- **(a) Resolution — use `10 x 10 x 20`.** The same J0126 data has been published as both
  `9x9x20` and `10x10x20` nm. Two independent checks pick `10x10x20`: the GT total path
  is 97.4 mm vs the paper's 97 mm (at `9x9x20` it is 91.1 mm, which does not match), and
  the ERL lands on 1.105 mm vs the reported 1.1 mm.
- **(b) Merge threshold — use `merge_threshold=0`.** This counts every merge, however
  small. A nonzero threshold ignores merges sharing fewer than that many skeleton nodes
  (dust merges) and so always reads higher — at `merge_threshold=50`, the repo default
  elsewhere, the same LUT gives 1.132 mm instead of 1.105 mm. State the threshold with
  any ERL you quote.

Assignment-zero (skeleton points landing on background) is 11948/500845 = 2.39%.
`gt ERL` depends only on the skeletons and the resolution, so at a fixed resolution it
is invariant across segmentations and thresholds — a useful check that two runs are
comparable.

To reproduce these rows, build the graph with the resolution applied
(`skel_to_erlgraph` takes ZYX order, so `10 x 10 x 20` xyz is `(20, 10, 10)`) and score
the same LUT; the node order is unchanged, only the edge lengths are scaled:

```python
from em_erl.io import load_skeletons
from em_erl.erl import skel_to_erlgraph
from em_erl.eval import load_node_segment_lut, compute_erl_score

skel = load_skeletons("test_50_skeletons.h5")          # numeric key sort == LUT node order
graph = skel_to_erlgraph(skel, skeleton_resolution=(20, 10, 10))   # nm, ZYX
lut = load_node_segment_lut("results/j0126/node_lut.h5")
score = compute_erl_score(graph, lut, None, merge_threshold=0)     # 0 = paper convention
score.compute_erl()
score.print_erl()
```

```
all skel
ERL     : 1105478.94
gt ERL  : 2114415.53
NERL    : 0.5228
#skel   : 50
```

Load the skeletons with `em_erl.io.load_skeletons` rather than raw HDF5 key iteration:
`load_skeletons` sorts keys numerically, while iterating the HDF5 file directly gives
`0, 1, 10, 11, ...`, which silently misaligns the LUT and produces a meaningless score
(0.053 instead of 0.538).

### Data download

```bash
# GT skeletons (also on HuggingFace, see "Data (processed examples)" below)
wget https://huggingface.co/datasets/pytc/zebrafinch-j0126/resolve/main/test_50_skeletons.h5
```

The FFN segmentation itself does **not** need downloading: the script streams it
from the public CloudVolume and writes a ~316 KB node LUT. Ship that LUT, not the
segmentation.

Optional — skip the sampling entirely by downloading the precomputed LUT
([`ffn_node_lut_test_50.h5`](https://huggingface.co/datasets/pytc/zebrafinch-j0126/resolve/main/ffn_node_lut_test_50.h5?download=true),
316 KB), which is the public FFN segmentation at mip0 sampled at every node of
`test_50_skeletons.h5`. It reproduces the table above and saves the ~3.6 GB of
CloudVolume egress:

```bash
wget https://huggingface.co/datasets/pytc/zebrafinch-j0126/resolve/main/ffn_node_lut_test_50.h5 \
     -O results/j0126/node_lut.h5
python examples/eval_j0126.py -g test_50_skeletons.h5 --lut results/j0126/node_lut.h5
```

The LUT is one segment id per skeleton node in `em_erl.io.load_skeletons` order
(HDF5 keys sorted **numerically**). Pair it with a graph built from the same loader —
indexing it with any other node order silently yields a wrong score rather than an error.

### Script

```bash
# first run: samples the public FFN CloudVolume (~3.6 GB one-time egress), writes the LUT
python examples/eval_j0126.py -g test_50_skeletons.h5 --lut results/j0126/node_lut.h5 -w 16

# later runs: score from the LUT, no CloudVolume access, ~0.1 s
python examples/eval_j0126.py -g test_50_skeletons.h5 --lut results/j0126/node_lut.h5
```

### Notes on conventions (read before comparing numbers)

Two independent choices change the value; a number is only meaningful with both stated.

1. **Aggregation — match `funlib.evaluate`.** `funlib/evaluate/run_length.py`
   computes `skeleton_erl = Σ_seg(correct_len² / skel_len)` and
   `ERL = Σ_skel (skel_len/total_len) · skeleton_erl`, i.e. `Σ(len·erl)/Σ(len)`.
   A perfect segmentation gives `erl_i = len_i`, so the normaliser is
   `Σ(len²)/Σ(len)` and

   ```
   NERL = Σ(len·erl) / Σ(len²)
   ```

   `em_erl.ERLScore.compute_erl` implements exactly this (`erl_pred = len*erl`,
   `erl_gt = len*len`, both divided by `total_len`), so `em_erl` is
   funlib-consistent and this is what the table above reports.

   Do **not** use `skeleton_erl.sum() / skeleton_len.sum()` (`Σerl/Σlen`). It drops
   the length weighting in the numerator and reads meaningfully higher — on the
   zebrafinch ABISS segmentation it gives 0.4135 where the funlib-consistent value
   is 0.3853. Numbers computed that way are not comparable to funlib or to this table.

2. **Units and merge threshold.** `eval_j0126.py` scores in voxel units
   (`skel_to_erlgraph` with no `skeleton_resolution`) at `merge_threshold=50`, which is
   the historical J0126 workflow but is *not* the paper's convention. To compare against
   the paper use `10 x 10 x 20` nm and `merge_threshold=0` — see "Reproduced number"
   above for both caveats and the measured effect of each. Always state the resolution
   and the threshold alongside any ERL.

Notes:
- Demonstrates `em_erl.evaluate_skeletons_cloudvolume`,
  `em_erl.sample_cloudvolume_lut`, and the node LUT save/load helpers.
- The workflow streams segment ids from the public FFN segmentation CloudVolume; no tile download is needed.
- The sampler fetches each occupied CloudVolume chunk exactly once. For the public unsharded `compressed_segmentation` layer, chunk `[128, 128, 64]` is atomic, so this is the GCS cost floor for generating a LUT at a given mip.
- Use `--lut` to make this a download-once workflow. The first run samples the segmentation and writes a compact node-to-segment LUT; later runs load that LUT and score without opening CloudVolume.
- Install the required extras first: `pip install -e ".[cloud,h5]"`.
- Once a LUT exists, scoring from `--lut` only needs the skeleton file plus normal HDF5 dependencies; `cloud-volume` is not imported or required.
- The score prints `ERL`, `gt ERL`, and `NERL` (= ERL / gt ERL).
- ERL is computed in voxel units to match the historical J0126 workflow (no
  resolution is applied), so the numbers are unaffected by the fact that the same
  j0126 data has been labeled inconsistently as `9x9x20 nm` and `10x10x20 nm`.
  Physical-unit ERL would depend on that assumption.

### Data (processed examples)
- GT skeletons: [test (50 neurons)](https://huggingface.co/datasets/pytc/zebrafinch-j0126/blob/main/test_50_skeletons.h5), [validation (12 neurons)](https://huggingface.co/datasets/pytc/zebrafinch-j0126/blob/main/valid_12_skeletons.h5)
- FFN segmentation (zip files): [part 1](https://huggingface.co/datasets/pytc/zebrafinch-j0126/resolve/main/ffn_agg_20-10-10_part1.zip?download=true), [part 2](https://huggingface.co/datasets/pytc/zebrafinch-j0126/resolve/main/ffn_agg_20-10-10_part2.zip?download=true)
- Optional training data: [33 subvolumes](https://huggingface.co/datasets/pytc/zebrafinch-j0126/blob/main/j0126-train-33vol.zip)
- Optional precomputed FFN node LUT (skip CloudVolume sampling): [ffn_node_lut_test_50.h5](https://huggingface.co/datasets/pytc/zebrafinch-j0126/resolve/main/ffn_node_lut_test_50.h5?download=true) (316 KB)

### Usage
```bash
python examples/eval_j0126.py \
    -g /projects/weilab/dataset/zebrafinch/test_50_skeletons.h5 \
    --lut results/j0126/node_lut.h5 \
    -w 16 \
    -o results/j0126/erl_score.pkl
```

The LUT and score default to `results/j0126/` (git-ignored). On the first run
with a missing `--lut` path, the script samples the public FFN CloudVolume and
saves the LUT; later runs reuse it without opening CloudVolume.

For the verified mip0 J0126 test skeletons this is a one-time egress cost of
about 3.6 GB. The saved LUT is about 4 MB raw, so it is the artifact to share
(e.g. on HuggingFace) so collaborators can score ERL without re-downloading the
FFN segmentation. Re-running the same command with the LUT present rebuilds
the ERL graph from `-g`, checks that `len(LUT) == graph.num_nodes`, and computes
the score without any CloudVolume access.

Generation options:
- `--mip INT` defaults to `0`, the faithful setting used for registration. A
  coarser mip downloads about 4x less data per level and is faster, but changes
  about 1.5% of sampled node labels: measured agreement was mip1==mip0 0.985
  and mip2==mip0 0.980. Use coarser mips only for cheaper/faster approximate
  ERL.
- `--cache-dir PATH` passes a local cache directory to CloudVolume while
  generating a missing LUT. It stores raw chunks locally to avoid re-download on
  a generation re-run, and is off by default. A saved LUT is still the preferred
  reusable artifact.

## Chunked volume ERL workflow (`eval_volume_chunk.py`)

`eval_volume_chunk.py` evaluates a segmentation stored as per-chunk HDF5 files:
map each tile to a partial node-to-segment LUT, reduce the partial LUTs into
`seg_lut_all.h5`, then score ERL from `gt_graph.npz`.

```bash
python examples/eval_volume_chunk.py --config examples/eval_volume_chunk.yaml --init-only
python examples/eval_volume_chunk.py --config examples/eval_volume_chunk.yaml --parallel 8
python examples/eval_volume_chunk.py --config examples/eval_volume_chunk.yaml --wait
python examples/eval_volume_chunk.py --config examples/eval_volume_chunk.yaml --reduce --score
```

Chunk ranges are template keys for `seg_path_format % (z, y, x)`. `factor` is a
length-3 zyx multiplier from key to voxel start, so each worker samples a tile
at `voxel_offset = [z, y, x] * factor`. For index-keyed files, use the chunk
shape as `factor`. For voxel-start-keyed axes, use `1` on that axis; the
j0126-style YAML uses `factor: [1, 2048, 2048]` because z keys are voxel starts
and y/x keys are tile indices.

Common modes:
- `--init-only` builds `gt_graph.npz` and writes shared node positions to
  `gt_vertices.h5`.
- `--chunk-index N` or `--chunk-range A-B` maps one or more chunks. A SLURM
  array task can provide `SLURM_ARRAY_TASK_ID`.
- `--parallel N` or `--local` maps the full grid with local multiprocessing.
- `--sbatch` emits and submits a SLURM array script under `slurm_outputs/`;
  `--wait` polls for all partial LUT files and fails after a stall timeout.
- `--reduce` combines partial LUTs; `--score` scores the combined LUT. They can
  be run together.

## Other examples

- `eval_volume.py` demonstrates `em_erl.compute_segment_lut`,
  `em_erl.compute_erl_score`, and `em_erl.ERLGraph`.
- `seg_to_graph.py` demonstrates `em_erl.seg_to_graph`
  (read volume -> skeletonize -> ERLGraph).
- `skel_to_graph.py` demonstrates `em_erl.skel_to_graph`
  (read a kimimaro skeleton -> ERLGraph).

Workflow outputs (LUTs, ERL score pickles, graphs) default to `results/`, which is
git-ignored.
