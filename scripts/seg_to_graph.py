import argparse
from em_erl.erl import skel_to_erlgraph
from em_erl.io import read_vol
from em_erl.skel import vol_to_skel


def run_seg_to_graph(seg_path, seg_resolution, length_threshold=0, num_thread=1):
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
    return skel_to_erlgraph(skel, length_threshold=length_threshold)


def parse_args():
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
        help="output npz file path. e.g., gt_graph.npz",
        default="gt_graph.npz",
    )
    parser.add_argument(
        "-l",
        "--length-threshold",
        type=int,
        help="throw away skeletons that are shorter than the threshold",
        default=0,
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


def main():
    # python scripts/seg_to_graph.py -s tests/data/vol_gt.h5 -r 30,30,30 -o tests/data/gt_graph.npz
    args = parse_args()

    # convert segment into a graph of its skeletons
    graph = run_seg_to_graph(
        args.seg_path, args.seg_resolution, args.length_threshold, args.num_thread
    )
    graph.print_info()
    graph.save_npz(args.output_path)


if __name__ == "__main__":
    main()
