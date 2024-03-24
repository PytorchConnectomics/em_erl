import os, argparse
from em_erl.networkx_lite import skel_to_lite
from em_util.io import read_vol, write_pkl
from em_util.skel import vol_to_skel


def test_skel_to_graph(skel_path, seg_resolution):
    print("Load skeleton")
    skel = read_vol(skel_path)
    return skel_to_lite(skel, seg_resolution)


def test_seg_to_graph(seg_path, seg_resolution, num_thread=1):
    """
    The function `test_gt_prep` takes in the paths to ground truth segmentation and its resolution.

    :param gt_stats_path: The path to the ground truth statistics file. This file contains information
    about the ground truth graph (vertex in physical unit) and resolution (used to convert node position to voxel)
    :param pred_seg_path: The `pred_seg_path` parameter is the file path to the predicted segmentation.
    It is the path to a file that contains the predicted segmentation data
    :param num_chunk: The parameter `num_chunk` is an optional parameter that specifies the number of
    chunks to divide the computation into. It is used in the function `compute_node_segment_lut_low_mem`
    to divide the computation of the node segment lookup table into smaller chunks, which can help
    reduce memory usage and improve performance, defaults to 1 (optional)
    """

    print("Load seg")
    seg = read_vol(seg_path)
    print("Compute skeleton")
    skel = vol_to_skel(seg, res=seg_resolution, num_thread=num_thread)

    print("Compute network")
    return skel_to_lite(skel, seg_resolution)


def get_arguments():
    """
    The function `get_arguments()` is used to parse command line arguments for the evaluation on AxonEM.
    :return: The function `get_arguments` returns the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Convert gt segmentation to graph of skeleton"
    )
    parser.add_argument(
        "-s",
        "--seg-path",
        type=str,
        help="path to the ground truth segmentation",
        default="",
    )
    parser.add_argument(
        "-s",
        "--skel-path",
        type=str,
        help="path to the ground truth skeleton",
        default="",
    )
    parser.add_argument(
        "-r",
        "--seg-resolution",
        type=str,
        help="resolution of the ground truth segmentation (zyx-order). e.g., 30,32,32",
        default="30,32,32",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        help="output pickle file path. e.g., gt_graph.pkl",
        default="gt_graph.pkl",
    )
    parser.add_argument(
        "-t",
        "--num-thread",
        type=int,
        help="number of threads for skeletonization",
        default=1,
    )
    args = parser.parse_args()
    assert (
        args.seg_path != "" or args.skel_path != ""
    ), "At least one of the paths to seg and skel is non-empty"
    args.seg_resolution = [float(x) for x in args.seg_resolution.split(",")]
    return args


if __name__ == "__main__":
    # python test_seg_to_graph.py -s db/30um_human/axon_release/gt_32nm.h5
    args = get_arguments()

    if os.path.exists(args.output_path):
        print(f"File {args.output_path} already exists.")
    else:
        # convert segment into a graph of its skeletons
        graph = test_seg_to_graph(
            args.seg_path, args.skel_path, args.seg_resolution, args.num_thread
        )
        write_pkl(args.output_path, graph)
