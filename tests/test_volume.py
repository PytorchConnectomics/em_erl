import argparse
from em_util.io import read_vol
from em_erl.erl import compute_segment_lut, compute_erl


def test_volume(
    pred_path,
    gt_path,
    gt_resolution,
    gt_mask_path="",
    merge_threshold=0,
    erl_intervals="",
):
    print("Load data")
    pred_seg = read_vol(pred_path)
    gt_mask = None if gt_mask_path == "" else read_vol(gt_mask_path)
    # graph: in physical unit
    gt_graph = read_vol(gt_path)

    print("Compute seg id for each gt skel node")
    # node position: need voxel unit
    node_voxel = (gt_graph.get_nodes()[:, 1:] // gt_resolution).astype(int)
    node_segment_lut, mask_segment_id = compute_segment_lut(
        pred_seg, node_voxel, gt_mask
    )

    print("Compute erl")
    scores = compute_erl(
        gt_graph=gt_graph,
        node_segment_lut=node_segment_lut,
        mask_segment_id=mask_segment_id,
        merge_threshold=merge_threshold,
        erl_intervals=erl_intervals,
        return_merge_split_stats=True,
    )

    print(f"{pred_path}\n-----")
    if erl_intervals is None:
        # overall evaluation
        print(f"pred ERL\t: {scores[0][0]:.2f}")
        print(f"gt ERL\t: {scores[0][1]:.2f}")
        print(f"#skel\t: {scores[0][2]:d}")
        print(f"errors\t: {scores[1]}")
    else:
        # break-down evaluation based on the skeleton length
        for i in range(len(erl_intervals) - 1):
            print(f"gt skel range: {erl_intervals[i]}-{erl_intervals[i+1]}")
            print(f"pred ERL\t: {scores[0][i, 0]:.2f}")
            print(f"gt ERL\t: {scores[0][i, 1]:.2f}")
            print(f"#skel\t: {scores[0][i, 2]:d}")
        print(f"errors\t: {scores[1]}")


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
        "-r",
        "--gt-resolution",
        type=str,
        help="resolution of the ground truth skeleton (zyx-order). e.g., 30,32,32",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--gt-mask-path",
        type=str,
        help="path to ground truth mask for false merge",
        default="",
    )
    parser.add_argument(
        "-t",
        "--merge-threshold",
        type=int,
        help="number of false merge voxels to classify a gt skel to have the false merge error",
        default=0,
    )
    parser.add_argument(
        "-i",
        "--erl-intervals",
        type=str,
        help="compute erl for each range. e.g., 0,5000,50000,150000",
        default="",
    )
    result = parser.parse_args()
    result.gt_resolution = [float(x) for x in result.gt_resolution.split(",")]
    result.erl_intervals = (
        [int(x) for x in result.erl_intervals.split(",")]
        if "," in result.erl_intervals
        else None
    )
    return result


if __name__ == "__main__":
    # python tests/test_volume.py -p tests/data/vol_pred.h5 -g tests/data/gt_graph.pkl -r 30,30,30 -m tests/data/vol_no-mask.h5
    # python tests/test_volume.py -p pni_seg_32nm.h5 -g axon_graph.pkl -r 30,32,32 -m axon_no-mask.h5
    args = get_arguments()
    test_volume(
        args.pred_path,
        args.gt_path,
        args.gt_resolution,
        args.gt_mask_path,
        args.merge_threshold,
        args.erl_intervals,
    )
