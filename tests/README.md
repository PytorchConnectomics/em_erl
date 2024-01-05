Reproducing FFN evaluation result (test_j0126.py)
----
[paper](https://www.nature.com/articles/s41592-018-0049-4), [data from the paper](https://storage.googleapis.com/j0126-nature-methods-data/GgwKmcKgrcoNxJccKuGIzRnQqfit9hnfK1ctZzNbnuU/README.txt)
- Data download (our processed version)
    - GT skeleton: [test (50 neurons)](https://huggingface.co/datasets/pytc/zebrafinch-j0126/blob/main/test_50_skeletons.h5), [validation (12 neurons)](https://huggingface.co/datasets/pytc/zebrafinch-j0126/blob/main/valid_12_skeletons.h5)
  
    - FFN segmentation: [part 1](), [part 2]()

    - (optional) training data: [33 subvolumes](https://huggingface.co/datasets/pytc/zebrafinch-j0126/blob/main/j0126-train-33vol.zip)
- Run script
```
# unzip FFN segmentation: XX/ffn_agg_20-10-10/
# gt skeleton file: YY/test_50_skeletons.h5
# step 0: convert gt skeleton into networkx-lite graph
python test_j0126.py --seg-folder XX/ffn_agg_20-10-10/ --gt-skeleton YY/test_50_skeletons.h5 --task 0

# step 1: compute seg id for each skeleton node for each segment tile (for parallellism, use `--job ${job_id},${job_num}`) 
python test_j0126.py --seg-folder XX/ffn_agg_20-10-10/ --gt-skeleton YY/test_50_skeletons.h5 --task 1

# step 2: combine results from step 1 into one file 
python test_j0126.py --seg-folder XX/ffn_agg_20-10-10/ --gt-skeleton YY/test_50_skeletons.h5 --task 2

# step 3: compute erl
python test_j0126.py --seg-folder XX/ffn_agg_20-10-10/ --gt-skeleton YY/test_50_skeletons.h5 --task 3
```
