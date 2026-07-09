"""Example CLI: build an ERL graph from a segmentation volume.

Demonstrates em_erl.seg_to_graph (read volume -> skeletonize -> ERLGraph).
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse

import em_erl


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a ground-truth segmentation into an ERL graph of skeletons"
    )
    parser.add_argument(
        "-s", "--seg-path", required=True, help="path to the ground truth segmentation"
    )
    parser.add_argument(
        "-r",
        "--seg-resolution",
        default="30,32,32",
        help="resolution of the segmentation (zyx-order). e.g., 30,32,32",
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
        "-t",
        "--num-thread",
        type=int,
        default=1,
        help="number of threads for skeletonization",
    )
    result = parser.parse_args()
    result.seg_resolution = [float(x) for x in result.seg_resolution.split(",")]
    return result


def main():
    # python examples/seg_to_graph.py -s tests/data/vol_gt.h5 -r 30,30,30 -o results/gt_graph.npz
    args = parse_args()
    graph = em_erl.seg_to_graph(
        args.seg_path, args.seg_resolution, args.length_threshold, args.num_thread
    )
    graph.print_info()
    graph.save_npz(args.output_path)


if __name__ == "__main__":
    main()
