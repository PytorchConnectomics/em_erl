import argparse
from em_util.io import read_vol, write_pkl
from em_erl.eval import compute_segment_lut, compute_erl_score
from em_erl.erl import ERLGraph


def test_volume(
    pred_path,
    gt_path,
    gt_resolution,
    gt_mask_path="",
    merge_threshold=0,
    erl_intervals=None,
    verbose=False,
):
    print("Load data")
    pred_seg = read_vol(pred_path)
    gt_mask = None if gt_mask_path == "" else read_vol(gt_mask_path)
    # erl graph: in physical unit
    gt_graph = ERLGraph(gt_path)

    print("Compute seg lookup table for gt skeletons")
    # node position: need voxel unit
    node_position = gt_graph.get_nodes_position(gt_resolution)
    node_segment_lut, mask_segment_id = compute_segment_lut(
        pred_seg, node_position, gt_mask
    )

    print("Compute erl")
    score = compute_erl_score(
        erl_graph=gt_graph,
        node_segment_lut=node_segment_lut,
        mask_segment_id=mask_segment_id,
        merge_threshold=merge_threshold,
        verbose=verbose,
    )
    score.compute_erl(erl_intervals)
    score.print_erl()
    return score


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
    parser.add_argument(
        "-v",
        "--verbose",
        type=lambda x: (str(x).lower() == "true"),
        help="store detailed info",
        default=False,
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        help="output pickle file path. e.g., erl_score.pkl",
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
    # python tests/test_volume.py -p tests/data/vol_pred.h5 -g tests/data/gt_graph.npz -r 30,30,30 -m tests/data/vol_no-mask.h5
    # python tests/test_volume.py -p pni_seg_32nm.h5 -g axon_graph.npz -r 30,32,32 -m axon_no-mask_erode2.h5 -i 5000,20000,50000,200000 -v True -t 30 -o axon_score.pkl
    args = get_arguments()
    erl_score = test_volume(
        args.pred_path,
        args.gt_path,
        args.gt_resolution,
        args.gt_mask_path,
        args.merge_threshold,
        args.erl_intervals,
        args.verbose,
    )
    if args.output_path != "":
        write_pkl(args.output_path, erl_score)
