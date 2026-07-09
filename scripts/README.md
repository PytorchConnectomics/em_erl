# Scripts

## J0126 CloudVolume workflow (`j0126_workflow.py`)

References: [FFN paper](https://www.nature.com/articles/s41592-018-0049-4), [J0126 data README](https://storage.googleapis.com/j0126-nature-methods-data/GgwKmcKgrcoNxJccKuGIzRnQqfit9hnfK1ctZzNbnuU/README.txt)

Notes:
- The workflow streams segment ids from the public FFN segmentation CloudVolume; no tile download is needed.
- The sampler fetches each occupied CloudVolume chunk exactly once. For the public unsharded `compressed_segmentation` layer, chunk `[128, 128, 64]` is atomic, so this is the GCS cost floor for generating a LUT at a given mip.
- Use `--lut` to make this a download-once workflow. The first run samples the segmentation and writes a compact node-to-segment LUT; later runs load that LUT and score without opening CloudVolume.
- Install the required extras first: `pip install -e ".[cloud,h5]"`.
- Once a LUT exists, scoring from `--lut` only needs the skeleton file plus normal HDF5 dependencies; `cloud-volume` is not imported or required.
- ERL is computed in voxel units to match the historical J0126 workflow.

### Data (processed examples)
- GT skeletons: [test (50 neurons)](https://huggingface.co/datasets/pytc/zebrafinch-j0126/blob/main/test_50_skeletons.h5), [validation (12 neurons)](https://huggingface.co/datasets/pytc/zebrafinch-j0126/blob/main/valid_12_skeletons.h5)
- FFN segmentation (zip files): [part 1](https://huggingface.co/datasets/pytc/zebrafinch-j0126/resolve/main/ffn_agg_20-10-10_part1.zip?download=true), [part 2](https://huggingface.co/datasets/pytc/zebrafinch-j0126/resolve/main/ffn_agg_20-10-10_part2.zip?download=true)
- Optional training data: [33 subvolumes](https://huggingface.co/datasets/pytc/zebrafinch-j0126/blob/main/j0126-train-33vol.zip)

### Usage
```bash
python scripts/j0126_workflow.py \
    -g /projects/weilab/dataset/zebrafinch/test_50_skeletons.h5 \
    --lut node_segment_lut.h5 \
    -w 16 \
    -o erl_score.pkl
```

On the first run with a missing `--lut` path, the script samples the public FFN
CloudVolume and saves the LUT:

```bash
python scripts/j0126_workflow.py \
    -g /projects/weilab/dataset/zebrafinch/test_50_skeletons.h5 \
    --lut shared/test_50_node_segment_lut.h5 \
    -w 16 \
    -o erl_score.pkl
```

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
