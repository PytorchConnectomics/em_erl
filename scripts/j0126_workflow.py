import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

from em_erl.eval import (
    compute_erl_score,
    print_skeleton_assignment_zero_stats,
)
from em_erl.erl import skel_to_erlgraph
from em_erl.io import read_vol, write_h5, write_pkl


DEFAULT_SEG_URL = (
    "gs://j0126-nature-methods-data/"
    "GgwKmcKgrcoNxJccKuGIzRnQqfit9hnfK1ctZzNbnuU/"
    "ffn_segmentation/"
)


def _skeleton_sort_key(key):
    try:
        return (0, int(key), str(key))
    except (TypeError, ValueError):
        return (1, str(key))


def _parse_skeleton_id(key, fallback):
    try:
        return int(key)
    except (TypeError, ValueError):
        return int(fallback)


def load_skeletons(gt_skeleton_path):
    skel_dict = {}
    with h5py.File(gt_skeleton_path, "r") as skeletons:
        keys = sorted(skeletons.keys(), key=_skeleton_sort_key)
        for i, key in enumerate(keys):
            group = skeletons[key]
            skel_dict[_parse_skeleton_id(key, i)] = SimpleNamespace(
                vertices=np.asarray(group["vertices"]),
                edges=np.asarray(group["edges"]),
            )
    return skel_dict


def normalize_seg_url(seg_url):
    url = str(seg_url).strip()
    prefix = "precomputed://"
    if url.startswith(prefix):
        url = url[len(prefix) :]
    if not url.startswith(("gs://", "https://")):
        raise ValueError(
            "segmentation URL must start with 'gs://', 'https://', "
            "or 'precomputed://' followed by one of those schemes"
        )
    return prefix + url


def open_seg_cloudvolume(seg_url, mip=0, cache_dir=""):
    try:
        from cloudvolume import CloudVolume
    except ImportError as exc:
        raise ImportError(
            "cloud-volume is required for J0126 CloudVolume sampling; "
            'install with: pip install -e ".[cloud,h5]"'
        ) from exc

    cache = str(cache_dir) if str(cache_dir) != "" else False
    return CloudVolume(
        normalize_seg_url(seg_url),
        mip=mip,
        cache=cache,
        use_https=True,
        fill_missing=True,
        bounded=False,
        progress=False,
    )


def _xyz_array(value, name):
    arr = np.asarray(value, dtype=np.int64).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"{name} must have exactly 3 xyz entries, got {value}")
    return arr


def _squeeze_cloudvolume_block(block):
    block = np.asarray(block)
    if block.ndim == 4:
        if block.shape[-1] != 1:
            raise ValueError(
                f"expected a single-channel CloudVolume block, got shape {block.shape}"
            )
        block = block[..., 0]
    if block.ndim != 3:
        raise ValueError(f"expected an xyz block, got shape {block.shape}")
    return block


def sample_cloudvolume_lut(cv, node_zyx, num_workers=16):
    node_zyx = np.asarray(node_zyx)
    if node_zyx.ndim != 2 or node_zyx.shape[1] != 3:
        raise ValueError(f"node_zyx must have shape [N, 3], got {node_zyx.shape}")

    lut = np.zeros(len(node_zyx), dtype=np.uint64)
    if len(node_zyx) == 0:
        return lut

    volume_size = _xyz_array(cv.volume_size, "volume_size")
    chunk_size = _xyz_array(cv.chunk_size, "chunk_size")
    if np.any(volume_size <= 0):
        raise ValueError(f"volume_size entries must be positive, got {volume_size}")
    if np.any(chunk_size <= 0):
        raise ValueError(f"chunk_size entries must be positive, got {chunk_size}")

    xyz = node_zyx[:, ::-1].astype(np.int64, copy=False)
    in_bounds = np.all((xyz >= 0) & (xyz < volume_size), axis=1)
    point_indices = np.flatnonzero(in_bounds)
    out_of_bounds = int(len(node_zyx) - len(point_indices))

    if len(point_indices) == 0:
        print(
            "Sampling 0 in-bounds skeleton nodes; "
            f"{out_of_bounds} out-of-bounds nodes left as segment 0"
        )
        return lut

    chunk_ids = xyz[point_indices] // chunk_size
    unique_chunks, inverse = np.unique(chunk_ids, axis=0, return_inverse=True)

    order = np.argsort(inverse, kind="stable")
    sorted_inverse = inverse[order]
    sorted_point_indices = point_indices[order]
    group_starts = np.r_[0, np.flatnonzero(np.diff(sorted_inverse)) + 1]
    group_ends = np.r_[group_starts[1:], len(sorted_point_indices)]
    groups = [
        sorted_point_indices[start:end] for start, end in zip(group_starts, group_ends)
    ]

    total_chunks = len(unique_chunks)
    print(
        f"Sampling {len(point_indices)}/{len(node_zyx)} in-bounds skeleton nodes "
        f"from {total_chunks} occupied chunks; "
        f"{out_of_bounds} out-of-bounds nodes left as segment 0"
    )

    def fetch_chunk(chunk_index):
        chunk_xyz = unique_chunks[chunk_index]
        indices = groups[chunk_index]
        start_xyz = chunk_xyz * chunk_size
        end_xyz = np.minimum(start_xyz + chunk_size, volume_size)
        block = _squeeze_cloudvolume_block(
            cv[
                int(start_xyz[0]) : int(end_xyz[0]),
                int(start_xyz[1]) : int(end_xyz[1]),
                int(start_xyz[2]) : int(end_xyz[2]),
            ]
        )
        local_xyz = xyz[indices] - start_xyz
        values = block[
            local_xyz[:, 0],
            local_xyz[:, 1],
            local_xyz[:, 2],
        ]
        return indices, np.asarray(values, dtype=np.uint64)

    max_workers = max(1, int(num_workers))
    progress_every = max(1, min(1000, total_chunks // 10 or 1))
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(fetch_chunk, chunk_index)
            for chunk_index in range(total_chunks)
        ]
        for future in as_completed(futures):
            indices, values = future.result()
            lut[indices] = values
            completed += 1
            if completed == total_chunks or completed % progress_every == 0:
                print(f"Fetched {completed}/{total_chunks} occupied chunks")

    return lut


def load_node_segment_lut(lut_path):
    """Load a saved node-to-segment LUT without importing CloudVolume."""
    return np.asarray(read_vol(lut_path), dtype=np.uint64)


def save_node_segment_lut(lut_path, node_segment_lut):
    lut = np.asarray(node_segment_lut, dtype=np.uint64)
    write_h5(lut_path, lut)
    print(f"Wrote node segment LUT: {lut_path}")
    return lut


def validate_node_segment_lut(node_segment_lut, graph, source="node_segment_lut"):
    lut = np.asarray(node_segment_lut, dtype=np.uint64)
    if lut.ndim != 1:
        raise ValueError(f"{source} must be a 1D array, got shape {lut.shape}")
    if len(lut) != graph.num_nodes:
        raise RuntimeError(
            f"{source} length does not match ERL graph node count: "
            f"{len(lut)} != {graph.num_nodes}. "
            "Regenerate the LUT with the same ground-truth skeleton file."
        )
    return lut


def score_graph_with_lut(
    graph,
    node_segment_lut,
    merge_threshold=50,
    output_path="",
    lut_source="node_segment_lut",
):
    node_segment_lut = validate_node_segment_lut(
        node_segment_lut,
        graph,
        source=lut_source,
    )

    print_skeleton_assignment_zero_stats(node_segment_lut)
    score = compute_erl_score(graph, node_segment_lut, None, merge_threshold)
    score.compute_erl()
    score.print_erl()

    if output_path != "":
        write_pkl(output_path, score)
        print(f"Wrote ERL score pickle: {output_path}")

    return score


def score_skeletons_with_lut(
    skel_dict,
    node_segment_lut,
    merge_threshold=50,
    output_path="",
    lut_source="node_segment_lut",
):
    print("Building ERL graph in voxel units")
    graph = skel_to_erlgraph(skel_dict)
    return score_graph_with_lut(
        graph,
        node_segment_lut,
        merge_threshold=merge_threshold,
        output_path=output_path,
        lut_source=lut_source,
    )


def run_j0126_eval(
    gt_skeleton_path,
    seg_url=DEFAULT_SEG_URL,
    merge_threshold=50,
    num_workers=16,
    output_path="",
    lut_path="",
    mip=0,
    cache_dir="",
):
    print(f"Loading ground-truth skeletons: {gt_skeleton_path}")
    skel_dict = load_skeletons(gt_skeleton_path)
    print(f"Loaded {len(skel_dict)} skeletons")

    print("Building ERL graph in voxel units")
    graph = skel_to_erlgraph(skel_dict)

    lut_source = "sampled node_segment_lut"
    lut_file = Path(lut_path) if str(lut_path) != "" else None
    if lut_file is not None and lut_file.exists():
        print(f"Loading node segment LUT: {lut_file}")
        node_segment_lut = load_node_segment_lut(lut_file)
        lut_source = f"loaded LUT {lut_file}"
    else:
        node_zyx = graph.get_nodes_position(None)
        print(f"Sampling segment ids for {graph.num_nodes} graph nodes")
        cv = open_seg_cloudvolume(seg_url, mip=mip, cache_dir=cache_dir)
        node_segment_lut = sample_cloudvolume_lut(cv, node_zyx, num_workers)
        node_segment_lut = validate_node_segment_lut(
            node_segment_lut,
            graph,
            source="sampled node_segment_lut",
        )
        if lut_file is not None:
            node_segment_lut = save_node_segment_lut(lut_file, node_segment_lut)
            lut_source = f"saved LUT {lut_file}"

    return score_graph_with_lut(
        graph,
        node_segment_lut,
        merge_threshold=merge_threshold,
        output_path=output_path,
        lut_source=lut_source,
    )


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
        default="",
        help=(
            "optional node-to-segment LUT HDF5 path; if it exists, load it and "
            "score without opening CloudVolume, otherwise sample and save it"
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
        default="",
        help="optional path for pickled ERLScore output",
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
