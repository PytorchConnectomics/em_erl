import argparse
from em_erl.erl import skel_to_erlgraph
from em_erl.io import read_vol


def run_skel_to_graph(skel_path, length_threshold=0, sample_ratio=1):
    # input skel: output from kimimaro
    print("Load skeleton")
    skel = read_vol(skel_path)
    print("Compute erl graph")
    return skel_to_erlgraph(
        skel, length_threshold=length_threshold, sample_ratio=sample_ratio
    )


def parse_args():
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
        "-r",
        "--sample-ratio",
        type=float,
        help="randomly sample skeletons by the ratio",
        default=1,
    )
    return parser.parse_args()


def main():
    # python scripts/skel_to_graph.py -s tests/data/gt_skel_kimimaro.pkl -o tests/data/gt_graph.npz
    # python scripts/skel_to_graph.py -s ./axon_32nm_skel.pkl -o axon_graph_r01.npz -l 5000 -r 0.1
    args = parse_args()
    # convert segment into a graph of its skeletons
    graph = run_skel_to_graph(args.skel_path, args.length_threshold, args.sample_ratio)
    graph.print_info()
    graph.save_npz(args.output_path)


if __name__ == "__main__":
    main()
