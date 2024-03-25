import argparse
import numpy as np
from em_util.io import read_vol
from em_erl.erl import compute_segment_lut, compute_erl
from em_erl.networkx_lite import skel_to_lite


def test_volume(pred_path, gt_path, gt_mask_path=""):
    pred_seg = read_vol(pred_path)
    gt_mask = None if gt_mask_path == "" else read_vol(gt_mask_path)
    # graph: in physical unit
    gt_graph, gt_res = read_vol(gt_path)
    # node position: need voxel unit
    node_voxel = (gt_graph.get_nodes()[:, 1:] // gt_res).astype(int)

    node_segment_lut, mask_id = compute_segment_lut(pred_seg, node_voxel, gt_mask)

    scores = compute_erl(
        gt_graph, node_segment_lut, mask_id, return_merge_split_stats=True
    )
    print(f"{pred_path}\n-----")
    print(f"ERL\t: {scores[0][0]:.2f}")
    print(f"gt ERL\t: {scores[0][1]:.2f}")
    print(f"#skel\t: {scores[0][2]:d}")


def get_arguments():
    """
    The `get_arguments` function is used to parse command line arguments for the ERL evaluation on small
    volume.
    :return: the parsed arguments from the command line.
    """
    parser = argparse.ArgumentParser(description="ERL evaluation on small volume")
    parser.add_argument(
        "-p",
        "--pred-path",
        type=str,
        help="path to the segmentation prediction",
        required=True,
    )
    parser.add_argument(
        "-g",
        "--gt-path",
        type=str,
        help="path to ground truth network-lite graph",
        default="",
    )
    parser.add_argument(
        "-m",
        "--gt-mask-path",
        type=str,
        help="path to ground truth mask for false merge",
        default="",
    )

    return parser.parse_args()


if __name__ == "__main__":
    # python tests/test_volume.py -p tests/data/vol_pred.h5 -g gt_graph.pkl -m tests/data/vol_no-mask.h5
    args = get_arguments()
    test_volume(args.pred_path, args.gt_path, args.gt_mask_path)
