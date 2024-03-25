import argparse
from em_erl.networkx_lite import skel_to_lite
from em_util.io import read_vol, write_pkl, vol_to_skel


def test_skel_to_graph(skel_path, skel_resolution):

    print("Load skeleton")
    skel = read_vol(skel_path)[0]
    print("Compute network")
    return skel_to_lite(skel, skel_resolution)


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
        "--skel-path",
        type=str,
        help="path to the ground truth skeleton",
        required=True,
    )
    parser.add_argument(
        "-r",
        "--skel-resolution",
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
    result = parser.parse_args()
    result.seg_resolution = [float(x) for x in result.seg_resolution.split(",")]
    return result


if __name__ == "__main__":
    # python test_seg_to_graph.py -s tests/data/gt_vol.h5 30,30,30
    args = get_arguments()
    # convert segment into a graph of its skeletons
    graph = test_skel_to_graph(args.skel_path, args.skel_resolution)
    write_pkl(args.output_path, graph)
