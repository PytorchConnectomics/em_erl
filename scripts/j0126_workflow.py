import argparse
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from em_erl.io import mkdir, read_vol, write_h5
from em_erl.eval import (
    combine_segment_lut_tile_zyx,
    compute_erl_score,
    compute_segment_lut_tile_zyx,
)
from em_erl.erl import ERLGraph, skel_to_erlgraph


J0126_Z_RANGE = 128 * np.arange(45)
J0126_Y_RANGE = np.arange(6)
J0126_X_RANGE = np.arange(6)


class J0126Paths:
    def __init__(self, folder):
        self.folder = Path(folder)

    @property
    def gt_vertices(self):
        return self.folder / "gt_vertices.h5"

    @property
    def gt_graph(self):
        return self.folder / "gt_graph.npz"

    @property
    def seg_lut_all(self):
        return self.folder / "seg_lut_all.h5"

    @property
    def seg_lut_template(self):
        return str(self.folder / "%04d" / "%d_%d.h5")

    @staticmethod
    def seg_pred_template(seg_folder):
        return str(Path(seg_folder) / "%04d" / "%d_%d.h5")


def _parse_skeleton_key(key, fallback):
    try:
        return int(key)
    except (TypeError, ValueError):
        return int(fallback)


def _parse_job_spec(job_spec):
    try:
        job_id_str, job_num_str = job_spec.split(",", 1)
        job_id = int(job_id_str)
        job_num = int(job_num_str)
    except Exception as exc:
        raise ValueError(f"Invalid job spec '{job_spec}', expected 'job_id,job_num'") from exc
    if job_num <= 0:
        raise ValueError("job_num must be > 0")
    if job_id < 0 or job_id >= job_num:
        raise ValueError("job_id must satisfy 0 <= job_id < job_num")
    return job_id, job_num


def _iter_j0126_tile_ranges(job=None):
    zran = J0126_Z_RANGE
    if job is not None:
        job_id, job_num = job
        zran = zran[job_id::job_num]
    return zran, J0126_Y_RANGE, J0126_X_RANGE


def prepare_gt(gt_skeleton_path, output_folder):
    paths = J0126Paths(output_folder)
    mkdir(str(paths.gt_vertices), "parent")

    if paths.gt_vertices.exists() and paths.gt_graph.exists():
        print(f"Files exist: {paths.gt_vertices}, {paths.gt_graph}")
        return

    with h5py.File(gt_skeleton_path, "r") as skeletons:
        keys = sorted(
            skeletons.keys(),
            key=lambda k: (_parse_skeleton_key(k, 0), str(k)),
        )

        skel_dict = {}
        vertices_all = []
        for i, key in enumerate(keys):
            group = skeletons[key]
            vertices = np.asarray(group["vertices"])
            edges = np.asarray(group["edges"])
            vertices_all.append(vertices)
            skel_dict[_parse_skeleton_key(key, i)] = SimpleNamespace(
                vertices=vertices,
                edges=edges,
            )

    if not paths.gt_vertices.exists():
        write_h5(str(paths.gt_vertices), np.vstack(vertices_all))

    if not paths.gt_graph.exists():
        graph = skel_to_erlgraph(skel_dict)
        graph.save_npz(str(paths.gt_graph))


def map_lut_tiles(seg_folder, output_folder, job_spec):
    paths = J0126Paths(output_folder)
    if not paths.gt_vertices.exists():
        raise FileNotFoundError(
            f"Missing {paths.gt_vertices}. Run 'prepare-gt' first."
        )

    job = _parse_job_spec(job_spec)
    zran, yran, xran = _iter_j0126_tile_ranges(job)
    pts = read_vol(str(paths.gt_vertices))
    mkdir(str(paths.folder))
    compute_segment_lut_tile_zyx(
        J0126Paths.seg_pred_template(seg_folder),
        zran,
        yran,
        xran,
        pts,
        paths.seg_lut_template,
    )


def reduce_lut_tiles(output_folder):
    paths = J0126Paths(output_folder)
    if paths.seg_lut_all.exists():
        print(f"File exists: {paths.seg_lut_all}")
        return

    zran, yran, xran = _iter_j0126_tile_ranges()
    _ = combine_segment_lut_tile_zyx(zran, yran, xran, paths.seg_lut_template, dry_run=True)
    out = combine_segment_lut_tile_zyx(zran, yran, xran, paths.seg_lut_template)
    write_h5(str(paths.seg_lut_all), out)


def score_erl(output_folder, merge_threshold):
    paths = J0126Paths(output_folder)
    if not paths.gt_graph.exists():
        raise FileNotFoundError(f"Missing {paths.gt_graph}. Run 'prepare-gt' first.")
    if not paths.seg_lut_all.exists():
        raise FileNotFoundError(
            f"Missing {paths.seg_lut_all}. Run 'reduce-lut' first."
        )

    gt_graph = ERLGraph.from_npz(str(paths.gt_graph))
    node_segment_lut = read_vol(str(paths.seg_lut_all))
    score = compute_erl_score(gt_graph, node_segment_lut, None, merge_threshold)
    score.compute_erl()
    score.print_erl()
    return score


def build_parser():
    parser = argparse.ArgumentParser(
        description="J0126 ERL workflow with readable subcommands"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_prepare = subparsers.add_parser(
        "prepare-gt",
        help="Export stacked GT vertices and build ERLGraph (.npz) from an HDF5 skeleton file",
    )
    p_prepare.add_argument(
        "-g",
        "--gt-skeleton",
        required=True,
        help="path to ground truth skeleton HDF5 file",
    )
    p_prepare.add_argument(
        "-o",
        "--output-folder",
        default="eval",
        help="output folder for gt_vertices.h5 and gt_graph.npz",
    )

    p_map = subparsers.add_parser(
        "map-lut",
        help="Map segmentation tile ids onto GT vertices for one parallel job shard",
    )
    p_map.add_argument(
        "-s",
        "--seg-folder",
        required=True,
        help="path to FFN segmentation prediction folder",
    )
    p_map.add_argument(
        "-o",
        "--output-folder",
        default="eval",
        help="evaluation output folder (must already contain gt_vertices.h5)",
    )
    p_map.add_argument(
        "-j",
        "--job",
        default="0,1",
        help="job_id,job_num shard spec (e.g. 0,8)",
    )

    p_reduce = subparsers.add_parser(
        "reduce-lut",
        help="Combine all per-tile LUT outputs into seg_lut_all.h5",
    )
    p_reduce.add_argument(
        "-o",
        "--output-folder",
        default="eval",
        help="evaluation output folder containing per-tile LUT files",
    )

    p_score = subparsers.add_parser(
        "score",
        help="Compute ERL from gt_graph.npz and seg_lut_all.h5",
    )
    p_score.add_argument(
        "-o",
        "--output-folder",
        default="eval",
        help="evaluation output folder",
    )
    p_score.add_argument(
        "-mt",
        "--merge-threshold",
        type=int,
        default=50,
        help="threshold number of voxels to be a false merge",
    )

    return parser


def main():
    args = build_parser().parse_args()

    if args.command == "prepare-gt":
        print("Step: prepare ground truth vertices and ERLGraph")
        prepare_gt(args.gt_skeleton, args.output_folder)
        return

    if args.command == "map-lut":
        print("Step: map segmentation LUT tiles")
        map_lut_tiles(args.seg_folder, args.output_folder, args.job)
        return

    if args.command == "reduce-lut":
        print("Step: reduce LUT tiles")
        reduce_lut_tiles(args.output_folder)
        return

    if args.command == "score":
        print("Step: compute ERL")
        score_erl(args.output_folder, args.merge_threshold)
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
