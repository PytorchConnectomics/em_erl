"""Example CLI: build an ERL graph from a kimimaro skeleton file.

Demonstrates em_erl.skel_to_graph (read skeleton -> ERLGraph).
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse

import em_erl


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a ground-truth skeleton into an ERL graph"
    )
    parser.add_argument(
        "-s", "--skel-path", required=True, help="path to the ground truth skeleton"
    )
    parser.add_argument(
        "-o",
        "--output-path",
        default="results/gt_graph.npz",
        help="output npz file path",
    )
    parser.add_argument(
        "-l",
        "--length-threshold",
        type=int,
        default=0,
        help="throw away skeletons that are shorter than the threshold",
    )
    parser.add_argument(
        "-r",
        "--sample-ratio",
        type=float,
        default=1,
        help="randomly sample skeletons by the ratio",
    )
    return parser.parse_args()


def main():
    # python examples/skel_to_graph.py -s tests/data/gt_skel_kimimaro.pkl -o results/gt_graph.npz
    args = parse_args()
    graph = em_erl.skel_to_graph(
        args.skel_path, args.length_threshold, args.sample_ratio
    )
    graph.print_info()
    graph.save_npz(args.output_path)


if __name__ == "__main__":
    main()
