# erl
Evaluation script for expected run length (ERL)


- Installation
```
pip install --editable .
```

- Ground-truth graph format
  - `ERLGraph` is stored as a compressed `.npz` file (current schema uses flat arrays with `edge_ptr`).
  - Load with `ERLGraph.from_npz(...)`. Legacy `.npz` graphs created by older versions are auto-converted on load.

- Key improvements over the original ERL code
  - `no-merge`: specify non-merging regions (for example, different semantic classes) with a mask, so false merges into forbidden regions are counted correctly.
  - merge tolerance (merge percentage / threshold): allow minor false-merge contacts instead of immediately forcing a skeleton score to `0`. In this implementation, this is controlled by `--merge-threshold` (voxel-count threshold).

# Example
<table>
  <tr align=center>
    <td>Ground Truth<br/> (2 seg)</td><td>Prediction<br/> (1 false merge and 1 false split)</td><td>No-merge Mask</td>
  </tr>
  <tr>
    <td> <img src="tests/figure/test_gt.png" width = 360px></td>
    <td><img src="tests/figure/test_pred.png" width = 360px></td>
    <td><img src="tests/figure/test_mask.png" width = 360px></td>
  </tr>
</table>


- GT: Each of the two axons is around 4&mu;m and the total ERL is 4.275&mu;m.
- Prediction: One predicted axon is falsely merged with a dendrite. The other two predicted segments are falsely split (around 2&mu;m each).
- Naive ERL evaluation (defined in the [FFN paper](https://www.nature.com/articles/s41592-018-0049-4))
  - `python scripts/volume_eval.py -p tests/data/vol_pred.h5 -g tests/data/gt_graph.npz -r 30,30,30`
  - ❌ The falsely merged segment is considered correct (ERL=4&mu;m), as the ground truth segments do not know the existence of other segments.
  - ✅ The ground truth axon matched with two falsely split segments has ERL=2&mu;m. It agrees with the intuition that 1 split error per 2&mu;m.
  - The total ERL=3.054&mu;m is overrated.
- Improved ERL evaluation with `no-merge` mask
  - `python scripts/volume_eval.py -p tests/data/vol_pred.h5 -g tests/data/gt_graph.npz -r 30,30,30 -m tests/data/vol_no-mask.h5`
  - ✅ The falsely merged segment is considered wrong, as it merges with the `no-merge` mask. The corresponding gt segment has ERL=0&mu;m.
  - ✅ The falsely split segment is the same as above.
  - The total ERL=1.176&mu;m is reasonable.
- Merge-tolerant false-merge handling (`--merge-threshold`)
  - `python scripts/volume_eval.py -p <pred.h5> -g <gt_graph.npz> -r 30,30,30 -m <no_merge_mask.h5> -t 30`
  - Small accidental merge contacts below the threshold are tolerated, instead of harshly zeroing the skeleton score for tiny false merges.

- Graph conversion helpers
  - From segmentation volume to ERL graph: `python scripts/seg_to_graph.py -s tests/data/vol_gt.h5 -r 30,30,30 -o tests/data/gt_graph.npz`
  - From kimimaro skeleton pickle to ERL graph: `python scripts/skel_to_graph.py -s tests/data/gt_skel_kimimaro.pkl -o tests/data/gt_graph.npz`

- J0126 tiled workflow (large-volume evaluation)
  - See `scripts/README.md` for the full large-volume J0126 workflow (`prepare-gt`, `map-lut`, `reduce-lut`, `score`).





# Change Log
---
- [Funkelab](https://github.com/funkelab): [Original implementation](https://github.com/funkelab/funlib.evaluate/blob/master/funlib/evaluate/run_length.py) with `networkx`, which requires much memory.
- [jasonkena](https://jasonkena.github.io/): Designed a `networkx-lite` class that only keeps relevant info, which is still costly to compute.
- current: Uses flat-array `ERLGraph` storage (node arrays + edge arrays + `edge_ptr`), precomputed edge lengths, and detailed error statistics.
