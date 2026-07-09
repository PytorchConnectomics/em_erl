"""Example CLI for em_erl CloudVolume skeleton evaluation helpers.

Demonstrates em_erl.evaluate_skeletons_cloudvolume and the node LUT helpers.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse

import em_erl


DEFAULT_SEG_URL = (
    "gs://j0126-nature-methods-data/"
    "GgwKmcKgrcoNxJccKuGIzRnQqfit9hnfK1ctZzNbnuU/"
    "ffn_segmentation/"
)


def run_j0126_eval(gt_skeleton_path, seg_url=DEFAULT_SEG_URL, **kw):
    return em_erl.evaluate_skeletons_cloudvolume(gt_skeleton_path, seg_url, **kw)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compute J0126 ERL from GT skeletons by sampling the public FFN "
            "segmentation CloudVolume once or by reusing a saved node LUT."
        )
    )
    parser.add_argument(
        "-g",
        "--gt-skeleton",
        required=True,
        help="path to ground truth skeleton HDF5 file",
    )
    parser.add_argument(
        "--seg-url",
        default=DEFAULT_SEG_URL,
        help="CloudVolume segmentation URL",
    )
    parser.add_argument(
        "-mt",
        "--merge-threshold",
        type=int,
        default=50,
        help="threshold number of voxels to be a false merge",
    )
    parser.add_argument(
        "-w",
        "--num-workers",
        type=int,
        default=16,
        help="number of CloudVolume chunk fetch worker threads",
    )
    parser.add_argument(
        "--lut",
        default="results/j0126_node_lut.h5",
        help=(
            "node-to-segment LUT HDF5 path (under results/ by default); if it "
            "exists, load it and score without opening CloudVolume, otherwise "
            "sample and save it. Pass '' to disable saving/reuse."
        ),
    )
    parser.add_argument(
        "--mip",
        type=int,
        default=0,
        help=(
            "CloudVolume mip for LUT generation (default 0, faithful). A coarser "
            "mip downloads about 4x less data per level and is faster, but changes "
            "about 1.5%% of sampled node labels (measured mip1==mip0 0.985, "
            "mip2==mip0 0.980), so ERL drifts slightly; use only for a "
            "cheaper/faster approximation."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default="",
        help=(
            "optional CloudVolume cache directory for LUT generation; stores raw "
            "chunks locally to avoid re-download on a generation re-run, off by "
            "default"
        ),
    )
    parser.add_argument(
        "-o",
        "--output-path",
        default="results/j0126_erl_score.pkl",
        help="path for pickled ERLScore output (under results/ by default)",
    )
    return parser


def main():
    args = build_parser().parse_args()
    run_j0126_eval(
        args.gt_skeleton,
        seg_url=args.seg_url,
        merge_threshold=args.merge_threshold,
        num_workers=args.num_workers,
        output_path=args.output_path,
        lut_path=args.lut,
        mip=args.mip,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
