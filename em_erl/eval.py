import os
import numpy as np
from .io import read_vol, write_h5, mkdir
from .erl import SkeletonScore, ERLScore
from .sampling import sample_segment_lut

# step 1: compute node_id-segment lookup table from predicted segmemtation and node positions
# step 2: compute the ERL from the lookup table and the gt graph
# graph: networkx by default. To save memory for grand-challenge evaluation, we use netowrkx_lite

def compute_segment_lut_tile(seg, pts, seg_oset=0, pts_oset=None):
    """Sample segmentation values for points inside one tile.

    Returns a boolean mask over `pts` and the sampled segment ids for points inside the tile.
    """
    pts = np.asarray(pts)
    if pts_oset is None:
        pts_oset = np.array([0, 0, 0], dtype=int)
    else:
        pts_oset = np.asarray(pts_oset, dtype=int)
    if pts.size == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=seg.dtype)
    sz = seg.shape
    ind = np.ones(pts.shape[0], dtype=bool)
    if pts_oset[0] > -1:
        ind &= (pts[:, 0] >= pts_oset[0]) & (pts[:, 0] < pts_oset[0] + sz[0])
    if pts_oset[1] > -1:
        ind &= (pts[:, 1] >= pts_oset[1]) & (pts[:, 1] < pts_oset[1] + sz[1])
    if pts_oset[2] > -1:
        ind &= (pts[:, 2] >= pts_oset[2]) & (pts[:, 2] < pts_oset[2] + sz[2])
    if not np.any(ind):
        return ind, np.zeros(0, dtype=seg.dtype)
    val = seg[
        pts[ind, 0] - pts_oset[0],
        pts[ind, 1] - pts_oset[1],
        pts[ind, 2] - pts_oset[2],
    ].copy()
    if seg_oset is not None and np.any(np.asarray(seg_oset) != 0):
        if np.isscalar(seg_oset):
            if seg_oset != 0:
                val[val > 0] += seg_oset
        else:
            raise ValueError("compute_segment_lut_tile only supports scalar seg_oset")
    return ind, val


def _write_tile_lut(output_file, point_count, selected_point_indices, values):
    ind = np.zeros(point_count, dtype=bool)
    if len(selected_point_indices) > 0:
        ind[np.asarray(selected_point_indices, dtype=np.int64)] = True
    write_h5(output_file, [ind, np.asarray(values)], ["ind", "val"])


def _empty_tile_lut(output_file, point_count, value_dtype=np.uint32):
    _write_tile_lut(
        output_file=output_file,
        point_count=point_count,
        selected_point_indices=np.zeros(0, dtype=np.int64),
        values=np.zeros(0, dtype=value_dtype),
    )


def _prepare_z_lookup(pts):
    pts = np.asarray(pts)
    order = np.argsort(pts[:, 0], kind="mergesort")
    z_sorted = pts[order, 0]
    return order, z_sorted


def _indices_for_z_range(order, z_sorted, z0, z1):
    lo = int(np.searchsorted(z_sorted, z0, side="left"))
    hi = int(np.searchsorted(z_sorted, z1, side="left"))
    return order[lo:hi]


def _resolve_seg_offset(seg_oset, idx=None):
    if isinstance(seg_oset, (list, tuple, np.ndarray)):
        if idx is None:
            raise ValueError("idx is required when seg_oset is a sequence")
        return seg_oset[idx]
    return seg_oset


def _prebin_points_by_yx(pts, factor):
    factor = np.asarray(factor, dtype=int)
    if factor.shape != (3,):
        raise ValueError("factor must be length-3")
    if factor[1] <= 0 or factor[2] <= 0:
        raise ValueError("factor[1] and factor[2] must be positive")
    y_bin = np.floor_divide(pts[:, 1], factor[1]).astype(int)
    x_bin = np.floor_divide(pts[:, 2], factor[2]).astype(int)
    bins = {}
    for idx, key in enumerate(zip(y_bin, x_bin)):
        bins.setdefault(key, []).append(idx)
    for key, indices in bins.items():
        arr = np.asarray(indices, dtype=np.int64)
        z_order = np.argsort(pts[arr, 0], kind="mergesort")
        bins[key] = arr[z_order]
    return bins

def compute_segment_lut_tile_z(
    seg_path_format, zran, pts, output_path_format, factor=1, dataset=None, seg_oset=0
):
    pts = np.asarray(pts)
    point_count = len(pts)
    z_order, z_sorted = _prepare_z_lookup(pts)
    for i, z in enumerate(zran):
        mkdir(output_path_format, "parent")
        sn = output_path_format % (z)
        if os.path.exists(sn):
            continue
        print(f"computing: {sn}")
        seg = read_vol(seg_path_format % (z), dataset)
        z0 = int(z * factor)
        z1 = z0 + int(seg.shape[0])
        candidate_idx = _indices_for_z_range(z_order, z_sorted, z0, z1)
        if len(candidate_idx) == 0:
            _empty_tile_lut(sn, point_count, value_dtype=seg.dtype)
            continue
        local_ind, val = compute_segment_lut_tile(
            seg,
            pts[candidate_idx],
            seg_oset=_resolve_seg_offset(seg_oset, i),
            pts_oset=[z0, -1, -1],
        )
        _write_tile_lut(sn, point_count, candidate_idx[local_ind], val)

def compute_segment_lut_tile_zyx(
    seg_path_format, zran, yran, xran, pts, output_path_format, factor=None, dataset=None, seg_oset=0
):
    if factor is None:
        factor = [1, 2048, 2048]
    factor = np.asarray(factor, dtype=int)
    pts = np.asarray(pts)
    point_count = len(pts)
    yx_bins = _prebin_points_by_yx(pts, factor)
    for z in zran:
        mkdir(output_path_format % (z, 0, 0), "parent")
        for y in yran:
            for x in xran:
                sn = output_path_format % (z, y, x)
                if os.path.exists(sn):
                    continue

                candidate_idx = yx_bins.get((int(y), int(x)))
                if candidate_idx is None or len(candidate_idx) == 0:
                    _empty_tile_lut(sn, point_count)
                    continue

                print(f"computing: {sn}")
                seg = read_vol(seg_path_format % (z, y, x), dataset)
                z0 = int(z * factor[0])
                z1 = z0 + int(seg.shape[0])
                z_vals = pts[candidate_idx, 0]
                lo = int(np.searchsorted(z_vals, z0, side="left"))
                hi = int(np.searchsorted(z_vals, z1, side="left"))
                if hi <= lo:
                    _empty_tile_lut(sn, point_count, value_dtype=seg.dtype)
                    continue
                candidate_z_idx = candidate_idx[lo:hi]
                local_ind, val = compute_segment_lut_tile(
                    seg,
                    pts[candidate_z_idx],
                    seg_oset=_resolve_seg_offset(seg_oset),
                    pts_oset=np.array([z, y, x]) * factor,
                )
                _write_tile_lut(sn, point_count, candidate_z_idx[local_ind], val)

def _combine_lut_from_files(output_files, dry_run=False):
    """Combine LUT results from a list of file paths."""
    out = None
    for output_file in output_files:
        if dry_run:
            if not os.path.exists(output_file):
                raise FileNotFoundError(f"File not exists: {output_file}")
        else:
            ind, val = read_vol(output_file)
            if out is None:
                out = np.zeros(ind.shape, dtype=val.dtype)
            out[ind] = val
    return out

def combine_segment_lut_tile(output_files, dry_run=False):
    return _combine_lut_from_files(output_files, dry_run)

def combine_segment_lut_tile_z(filename_format, zran, dry_run=False):
    files = [filename_format % z for z in zran]
    return _combine_lut_from_files(files, dry_run)

def combine_segment_lut_tile_zyx(zran, yran, xran, output_path_format, dry_run=False):
    files = [output_path_format % (z, y, x) for z in zran for y in yran for x in xran]
    return _combine_lut_from_files(files, dry_run)


def compute_segment_lut(
    segment,
    node_position,
    mask=None,
    chunk_num=1,
    data_type=np.uint32,
    segment_dataset=None,
    mask_dataset=None,
):
    """
    The function `compute_segment_lut` is a low memory version of a lookup table
    computation for node segments in a 3D volume.

    :param node_position: A numpy array containing the coordinates of each node. The shape of the array
    is (N, 3), where N is the number of nodes and each row represents the (z, y, x) coordinates of a
    node
    :param segment: either a 3D volume or a string representing the
    name of a file containing segment data.
    :param chunk_num: The parameter `chunk_num` is the number of chunks into which the volume is divided
    for reading. It is used in the `read_vol` function to specify which chunk to read, defaults to 1
    (optional)
    :param data_type: The parameter `data_type` is the data type of the array used to store the node segment
    lookup table. In this case, it is set to `np.uint32`, which means the array will store unsigned
    32-bit integers
    :return: a list of numpy arrays, where each array represents the node segment lookup table for a
    specific segment.
    """
    return sample_segment_lut(
        segment=segment,
        node_position_zyx=node_position,
        mask=mask,
        chunk_num=chunk_num,
        data_type=data_type,
        segment_dataset=segment_dataset,
        mask_dataset=mask_dataset,
    )


def _try_pack_pair_keys(left, right):
    left_i = np.asarray(left, dtype=np.int64)
    right_i = np.asarray(right, dtype=np.int64)
    if left_i.shape != right_i.shape:
        raise ValueError("Pair arrays must have the same shape")
    if left_i.size == 0:
        return np.zeros(0, dtype=np.int64), 1
    if left_i.min() < 0 or right_i.min() < 0:
        return None, None

    right_max = int(right_i.max())
    left_max = int(left_i.max())
    radix = right_max + 1
    max_int64 = np.iinfo(np.int64).max
    if radix <= 0 or left_max > (max_int64 - right_max) // radix:
        return None, None
    return left_i * radix + right_i, radix


def _unique_pair_counts(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    if left.size == 0:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=right.dtype if right.ndim > 0 else np.int64),
            np.zeros(0, dtype=np.int64),
        )

    keys, radix = _try_pack_pair_keys(left, right)
    if keys is not None:
        uniq, counts = np.unique(keys, return_counts=True)
        left_u = (uniq // radix).astype(np.int64, copy=False)
        right_u = (uniq % radix).astype(right.dtype, copy=False)
        return left_u, right_u, counts.astype(np.int64, copy=False)

    pairs, counts = np.unique(
        np.column_stack([left, right]),
        axis=0,
        return_counts=True,
    )
    return (
        pairs[:, 0].astype(np.int64, copy=False),
        pairs[:, 1].astype(right.dtype, copy=False),
        counts.astype(np.int64, copy=False),
    )


def _sum_weights_by_pair(left, right, weights):
    left = np.asarray(left)
    right = np.asarray(right)
    weights = np.asarray(weights, dtype=np.float64)
    if left.size == 0:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=right.dtype if right.ndim > 0 else np.int64),
            np.zeros(0, dtype=np.float64),
        )

    keys, radix = _try_pack_pair_keys(left, right)
    if keys is not None:
        uniq, inv = np.unique(keys, return_inverse=True)
        sums = np.bincount(inv, weights=weights)
        left_u = (uniq // radix).astype(np.int64, copy=False)
        right_u = (uniq % radix).astype(right.dtype, copy=False)
        return left_u, right_u, sums

    pairs, inv = np.unique(
        np.column_stack([left, right]),
        axis=0,
        return_inverse=True,
    )
    sums = np.bincount(inv, weights=weights)
    return (
        pairs[:, 0].astype(np.int64, copy=False),
        pairs[:, 1].astype(right.dtype, copy=False),
        sums,
    )


def _group_ptr_from_sorted_left(left_sorted, group_count):
    if group_count == 0:
        return np.zeros(1, dtype=np.int64)
    left_sorted = np.asarray(left_sorted, dtype=np.int64)
    search = np.arange(group_count + 1, dtype=np.int64)
    return np.searchsorted(left_sorted, search, side="left")


def compute_erl_score(
    erl_graph,
    node_segment_lut,
    mask_segment_id,
    merge_threshold,
    verbose=False,
):
    node_segment_lut = np.asarray(node_segment_lut)
    erl_score = ERLScore(erl_graph.skeleton_id, erl_graph.skeleton_len, verbose)
    num_skeleton = len(erl_graph.skeleton_id)
    erl_score.skeleton_erl = -np.ones(num_skeleton, dtype=np.float64)
    if verbose:
        erl_score.skeleton_score = [SkeletonScore() for _ in range(num_skeleton)]

    # 1. find false merges among gt skeletons (vectorized pair counting)
    pair_skel, pair_seg, pair_count = _unique_pair_counts(
        erl_graph.node_skeleton_index,
        node_segment_lut,
    )
    big_pair_mask = pair_count >= merge_threshold
    big_pair_skel = pair_skel[big_pair_mask]
    big_pair_seg = pair_seg[big_pair_mask]
    big_pair_count = pair_count[big_pair_mask]

    seg_big, seg_big_counts = np.unique(big_pair_seg, return_counts=True)
    merged_segments = seg_big[seg_big_counts > 1]
    merged_skeletons = np.unique(pair_skel[np.isin(pair_seg, merged_segments)]).astype(int)

    erl_score.skeleton_erl[merged_skeletons] = 0

    # 2. find false merges with mask
    merged_mask = None
    merged_mask_skeletons = np.zeros(0, dtype=int)
    if mask_segment_id is not None:
        mask_segment_id = np.asarray(mask_segment_id)
        mask_id, mask_count = np.unique(mask_segment_id, return_counts=True)
        merged_mask = mask_id[mask_count > merge_threshold]
        if merged_mask.size > 0 and big_pair_seg.size > 0:
            merged_mask_skeletons = np.unique(
                big_pair_skel[np.isin(big_pair_seg, merged_mask)]
            ).astype(int)
            erl_score.skeleton_erl[merged_mask_skeletons] = 0

    # 3. compute ERL for skeletons without merge errors (vectorized edge grouping)
    active_skeleton_mask = erl_score.skeleton_erl < 0
    erl_score.skeleton_erl[active_skeleton_mask] = 0

    edge_skel = erl_graph.edge_skeleton_index().astype(np.int64, copy=False)
    active_edge_mask = (
        np.zeros(0, dtype=bool)
        if edge_skel.size == 0
        else active_skeleton_mask[edge_skel]
    )

    edge_skel_active = edge_skel[active_edge_mask]
    edge_u_active = erl_graph.edge_u[active_edge_mask].astype(np.int64, copy=False)
    edge_v_active = erl_graph.edge_v[active_edge_mask].astype(np.int64, copy=False)
    edge_len_active = erl_graph.edge_len[active_edge_mask].astype(np.float64, copy=False)

    if edge_u_active.size > 0:
        seg_u_active = node_segment_lut[edge_u_active]
        seg_v_active = node_segment_lut[edge_v_active]
        correct_edge_mask = seg_u_active == seg_v_active
    else:
        seg_u_active = np.zeros(0, dtype=node_segment_lut.dtype)
        seg_v_active = np.zeros(0, dtype=node_segment_lut.dtype)
        correct_edge_mask = np.zeros(0, dtype=bool)

    if np.any(correct_edge_mask):
        correct_pair_skel, correct_pair_seg, correct_pair_len = _sum_weights_by_pair(
            edge_skel_active[correct_edge_mask],
            seg_u_active[correct_edge_mask],
            edge_len_active[correct_edge_mask],
        )
        erl_terms = (
            correct_pair_len * correct_pair_len / erl_graph.skeleton_len[correct_pair_skel]
        )
        erl_score.skeleton_erl += np.bincount(
            correct_pair_skel,
            weights=erl_terms,
            minlength=num_skeleton,
        )
    else:
        correct_pair_skel = np.zeros(0, dtype=np.int64)
        correct_pair_seg = np.zeros(0, dtype=node_segment_lut.dtype)
        correct_pair_len = np.zeros(0, dtype=np.float64)

    # 4. verbose diagnostics (secondary pass; not on the hot path)
    if verbose:
        erl_score.merged_mask = merged_mask

        if merged_segments.size > 0:
            for seg in merged_segments:
                erl_score.merged_seg[seg] = big_pair_skel[big_pair_seg == seg]
            merged_seg_lookup = set(np.asarray(merged_segments).tolist())
            for skel in merged_skeletons:
                mask = big_pair_skel == skel
                segs = big_pair_seg[mask]
                counts = big_pair_count[mask]
                keep = np.array([s in merged_seg_lookup for s in segs], dtype=bool)
                erl_score.skeleton_score[skel].merged_seg = segs[keep]
                erl_score.skeleton_score[skel].merged_seg_num = counts[keep].astype(int).tolist()

        if merged_mask is not None and len(merged_mask_skeletons) > 0:
            merged_mask_lookup = set(np.asarray(merged_mask).tolist())
            for skel in merged_mask_skeletons:
                mask = big_pair_skel == skel
                segs = big_pair_seg[mask]
                counts = big_pair_count[mask]
                keep = np.array([s in merged_mask_lookup for s in segs], dtype=bool)
                erl_score.skeleton_score[skel].merged_mask = segs[keep]
                erl_score.skeleton_score[skel].merged_mask_num = counts[keep].astype(int).tolist()

        active_idx = np.where(active_skeleton_mask)[0]
        if edge_skel_active.size > 0:
            omitted_edge = np.bincount(
                edge_skel_active,
                weights=((seg_u_active == 0) | (seg_v_active == 0)).astype(np.int64),
                minlength=num_skeleton,
            ).astype(int, copy=False)
            split_edge = np.bincount(
                edge_skel_active,
                weights=(seg_u_active != seg_v_active).astype(np.int64),
                minlength=num_skeleton,
            ).astype(int, copy=False)
        else:
            omitted_edge = np.zeros(num_skeleton, dtype=int)
            split_edge = np.zeros(num_skeleton, dtype=int)

        correct_ptr = _group_ptr_from_sorted_left(correct_pair_skel, num_skeleton)
        for skel in active_idx:
            score = erl_score.skeleton_score[skel]
            score.omitted = int(omitted_edge[skel])
            score.split = int(split_edge[skel])
            start = int(correct_ptr[skel])
            end = int(correct_ptr[skel + 1])
            score.correct_seg = correct_pair_seg[start:end]
            score.correct_len = correct_pair_len[start:end]

    return erl_score
