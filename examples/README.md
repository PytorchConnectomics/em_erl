# Examples

These examples run directly from a source checkout because each script adds the
repository root to `sys.path`. They also work after `pip install -e .`.

## J0126 CloudVolume workflow (`eval_j0126.py`)

References: [FFN paper](https://www.nature.com/articles/s41592-018-0049-4), [J0126 data README](https://storage.googleapis.com/j0126-nature-methods-data/GgwKmcKgrcoNxJccKuGIzRnQqfit9hnfK1ctZzNbnuU/README.txt)

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

### Usage
```bash
python examples/eval_j0126.py \
    -g /projects/weilab/dataset/zebrafinch/test_50_skeletons.h5 \
    --lut results/j0126_node_lut.h5 \
    -w 16 \
    -o results/j0126_erl_score.pkl
```

The LUT and score default to `results/` (git-ignored). On the first run with a
missing `--lut` path, the script samples the public FFN CloudVolume and saves the
LUT; later runs reuse it without opening CloudVolume.

For the verified mip0 J0126 test skeletons this is a one-time egress cost of
about 3.6 GB. The saved LUT is about 4 MB raw, so it is the artifact to share
with collaborators. Re-running the same command with the LUT present rebuilds
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

## Other examples

- `eval_volume.py` demonstrates `em_erl.compute_segment_lut`,
  `em_erl.compute_erl_score`, and `em_erl.ERLGraph`.
- `seg_to_graph.py` demonstrates `em_erl.seg_to_graph`
  (read volume -> skeletonize -> ERLGraph).
- `skel_to_graph.py` demonstrates `em_erl.skel_to_graph`
  (read a kimimaro skeleton -> ERLGraph).

Workflow outputs (LUTs, ERL score pickles, graphs) default to `results/`, which is
git-ignored.
