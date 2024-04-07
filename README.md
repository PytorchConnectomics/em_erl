# erl
Evaluation script for expected run length (ERL)


- Installation
```
pip install --editable .
```

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
- Naive ERL evaluation
  - `python tests/test_volume.py -p tests/data/vol_pred.h5 -g tests/data/gt_graph.npz -r 30,30,30`
  - ❌ The falsely merged segment is considered correct (ERL=4&mu;m), as the ground truth segments do not know the existence of other segments.
  - ✅ The ground truth axon matched with two falsely split segments has ERL=2&mu;m. It agrees with the intuition that 1 split error per 2&mu;m.
  - The total ERL=3.054&mu;m is overrated.
- Improved ERL evaluation with `no-merge` mask
  - `python tests/test_volume.py -p tests/data/vol_pred.h5 -g tests/data/gt_graph.npz -r 30,30,30 -m tests/data/vol_no-mask.h5`
  - ✅ The falsely merged segment is considered wrong, as it merges with the `no-merge` mask. The corresponding gt segment has ERL=0&mu;m.
  - ✅ The falsely split segment is the same as above.
  - The total ERL=1.176&mu;m is reasonable.





# Change Log
---
- [Funkelab](https://github.com/funkelab): [Original implementation](https://github.com/funkelab/funlib.evaluate/blob/master/funlib/evaluate/run_length.py) with `networkx`, which requires much memory.
- [jasonkena](https://jasonkena.github.io/): Designed a `networkx-lite` class that only keeps relevant info, which is still costly to compute.
- current: Used `matrix` representation for skeleton-node, precompute edge length, and record error information for in-depth analysis.
