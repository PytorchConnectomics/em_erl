import argparse
from em_erl.networkx_lite import skel_to_lite
from em_util.io import read_vol, write_pkl


def test_skel_to_graph(skel_path):
    # input skel: output from kimimaro
    print("Load skeleton")
    skel = read_vol(skel_path)
    print("Compute network")
    return skel_to_lite(skel)


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
        "-o",
        "--output-path",
        type=str,
        help="output pickle file path. e.g., gt_graph.pkl",
        default="gt_graph.pkl",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # python tests/test_skel_to_graph.py -s tests/data/gt_skel_kimimaro.pkl -o tests/data/gt_graph.pkl
    args = get_arguments()
    # convert segment into a graph of its skeletons
    graph = test_skel_to_graph(args.skel_path)
    write_pkl(args.output_path, graph)
