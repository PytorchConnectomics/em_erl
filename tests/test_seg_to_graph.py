import argparse
from em_erl.networkx_lite import skel_to_lite
from em_util.io import read_vol, write_pkl, vol_to_skel


def test_seg_to_graph(seg_path, seg_resolution, num_thread=1):
    """
    Test the conversion of a segmentation volume to a graph representation.

    Args:
        seg_path (str): The file path to the segmentation volume.
        seg_resolution (float): The resolution of the segmentation volume.
        num_thread (int, optional): Number of threads for computation. Defaults to 1.

    Returns:
        networkx.Graph: A graph representation of the segmentation volume.
    """

    print("Load seg")
    seg = read_vol(seg_path)
    print("Compute skeleton")
    # skeleton nodes: in physical unit
    skel = vol_to_skel(seg, res=seg_resolution, num_thread=num_thread)
    print("Compute network")
    return skel_to_lite(skel)


def get_arguments():
    """
    Get command line arguments for converting ground truth segmentation to a graph of skeleton.

    Returns:
        argparse.Namespace: Parsed command line arguments including seg_path, seg_resolution, output_path, and num_thread.
    """
    parser = argparse.ArgumentParser(
        description="Convert gt segmentation to graph of skeleton"
    )
    parser.add_argument(
        "-s",
        "--seg-path",
        type=str,
        help="path to the ground truth segmentation",
        required=True,
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
    result = parser.parse_args()
    result.seg_resolution = [float(x) for x in result.seg_resolution.split(",")]
    return result


if __name__ == "__main__":
    # python tests/test_seg_to_graph.py -s tests/data/vol_gt.h5 -r 30,30,30 -o tests/data/gt_graph.pkl
    args = get_arguments()

    # convert segment into a graph of its skeletons
    graph = test_seg_to_graph(args.seg_path, args.seg_resolution, args.num_thread)
    write_pkl(args.output_path, graph)
