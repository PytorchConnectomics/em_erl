# erl
Evaluation script for expected run length (ERL)


- Installation
```
# create a new environment
pip install --editable .
```

# Example
<table>
  <tr align=center>
    <td>ground truth<br/> (2 seg)</td><td>prediction<br/> (1 false merge and 1 false split)</td><td>no-merge mask</td>
  </tr>
  <tr>
    <td> <img src="tests/figure/test_gt.png" width = 360px></td>
    <td><img src="tests/figure/test_pred.png" width = 360px></td>
    <td><img src="tests/figure/test_mask.png" width = 360px></td>
  </tr>
</table>



# Acknowledgement
---
- [Funkelab](https://github.com/funkelab): The main functionality is adapted from [this file](https://github.com/funkelab/funlib.evaluate/blob/master/funlib/evaluate/run_length.py)
- [jasonkena](https://jasonkena.github.io/): Reduce the memory usage by replacing the networkx object with a self-defined networkx-lite object.
