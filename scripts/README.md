# Scripts

## J0126 tiled workflow (`j0126_workflow.py`)

References: [FFN paper](https://www.nature.com/articles/s41592-018-0049-4), [J0126 data README](https://storage.googleapis.com/j0126-nature-methods-data/GgwKmcKgrcoNxJccKuGIzRnQqfit9hnfK1ctZzNbnuU/README.txt)

Notes:
- The workflow builds a ground-truth `ERLGraph` as `gt_graph.npz` (flat-array schema).
- Large-volume LUT mapping is split into `map-lut` and `reduce-lut` for parallel execution.

### Data (processed examples)
- GT skeletons: [test (50 neurons)](https://huggingface.co/datasets/pytc/zebrafinch-j0126/blob/main/test_50_skeletons.h5), [validation (12 neurons)](https://huggingface.co/datasets/pytc/zebrafinch-j0126/blob/main/valid_12_skeletons.h5)
- FFN segmentation (zip files): [part 1](https://huggingface.co/datasets/pytc/zebrafinch-j0126/resolve/main/ffn_agg_20-10-10_part1.zip?download=true), [part 2](https://huggingface.co/datasets/pytc/zebrafinch-j0126/resolve/main/ffn_agg_20-10-10_part2.zip?download=true)
- Optional training data: [33 subvolumes](https://huggingface.co/datasets/pytc/zebrafinch-j0126/blob/main/j0126-train-33vol.zip)

### Usage
```bash
# Example paths:
#   segmentation tiles: XX/ffn_agg_20-10-10/
#   gt skeleton file:   YY/test_50_skeletons.h5
#   evaluation folder:  ZZ/j0126_eval/

# 1) Prepare GT artifacts (exports stacked vertices + builds gt_graph.npz)
python scripts/j0126_workflow.py prepare-gt -g YY/test_50_skeletons.h5 -o ZZ/j0126_eval

# 2) Map segmentation tile ids to GT vertices (parallelizable by shard)
# run in parallel for shards 1,8 ... 7,8
python scripts/j0126_workflow.py map-lut -s XX/ffn_agg_20-10-10 -o ZZ/j0126_eval -j 0,8



# 3) Reduce all tile LUT outputs into seg_lut_all.h5
python scripts/j0126_workflow.py reduce-lut -o ZZ/j0126_eval

# 4) Compute ERL
python scripts/j0126_workflow.py score -o ZZ/j0126_eval
```
