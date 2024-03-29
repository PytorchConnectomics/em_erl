import os
import numpy as np
from em_util.io import read_vol, write_h5, mkdir
from .erl import SkeletonScore, ERLScore

# step 1: compute node_id-segment lookup table from predicted segmemtation and node positions
# step 2: compute the ERL from the lookup table and the gt graph
# graph: networkx by default. To save memory for grand-challenge evaluation, we use netowrkx_lite


def compute_segment_lut_tile(
    seg_path_format, zran, yran, xran, pts, output_path_format, factor=[1, 2048, 2048]
):
    for z in zran:
        mkdir(output_path_format % (z, 0, 0), "parent")
        for y in yran:
            for x in xran:
                sn = output_path_format % (z, y, x)
                if not os.path.exists(sn):
                    print(f"computing: {sn}")
                    seg = read_vol(seg_path_format % (z, y, x))
                    sz = seg.shape
                    ind = (pts[:, 0] >= z * factor[0]) * (
                        pts[:, 0] < z * factor[0] + sz[0]
                    )
                    ind = (
                        ind
                        * (pts[:, 1] >= y * factor[1])
                        * (pts[:, 1] < y * factor[1] + sz[1])
                    )
                    ind = (
                        ind
                        * (pts[:, 2] >= x * factor[2])
                        * (pts[:, 2] < x * factor[2] + sz[2])
                    )
                    val = seg[
                        pts[ind, 0] - z * factor[0],
                        pts[ind, 1] - y * factor[1],
                        pts[ind, 2] - x * factor[2],
                    ]
                    write_h5(sn, [ind, val], ["ind", "val"])


def compute_segment_lut_tile_combine(
    zran, yran, xran, output_path_format, dry_run=False
):
    out = None
    for z in zran:
        for y in yran:
            for x in xran:
                sn = output_path_format % (z, y, x)
                if dry_run:
                    if not os.path.exists(sn):
                        raise f"File not exists: {sn}"
                else:
                    ind, val = read_vol(sn)
                    if out is None:
                        out = np.zeros(ind.shape).astype(val.dtype)
                    out[ind] = val
    return out


def compute_segment_lut(
    segment, node_position, mask=None, chunk_num=1, data_type=np.uint32
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
    if not isinstance(segment, str):
        # load the whole segment
        node_lut = segment[
            node_position[:, 0], node_position[:, 1], node_position[:, 2]
        ]
        mask_id = []
        if mask is not None:
            if isinstance(mask, str):
                mask = read_vol(mask)
            mask_id = segment[mask > 0]
    else:
        # read segment by chunk (when memory is limited)
        assert ".h5" in segment
        node_lut = np.zeros(node_position.shape[0], data_type)
        mask_id = [[]] * chunk_num
        start_z = 0
        for chunk_id in range(chunk_num):
            seg = read_vol(segment, None, chunk_id, chunk_num)
            last_z = start_z + seg.shape[0]
            ind = (node_position[:, 0] >= start_z) * (node_position[:, 0] < last_z)
            pts = node_position[ind]
            node_lut[ind] = seg[pts[:, 0] - start_z, pts[:, 1], pts[:, 2]]
            if mask is not None:
                if isinstance(mask, str):
                    mask_z = read_vol(mask, None, chunk_id, chunk_num)
                else:
                    mask_z = mask[start_z:last_z]
                mask_id[chunk_id] = seg[mask_z > 0]
            start_z = last_z
        if mask is not None:
            # remove irrelevant seg ids (not used by nodes)
            node_lut_unique = np.unique(node_lut)
            for chunk_id in range(chunk_num):
                mask_id[chunk_id] = mask_id[chunk_id][
                    np.in1d(mask_id[chunk_id], node_lut_unique)
                ]
        mask_id = np.concatenate(mask_id)
    return node_lut, mask_id


def compute_erl_score(
    erl_graph,
    node_segment_lut,
    mask_segment_id,
    merge_threshold,
    verbose=False,
):
    erl_score = ERLScore(erl_graph.skeleton_id, erl_graph.skeleton_len, verbose)
    num_skeleton = len(erl_graph.skeleton_id)
    erl_score.skeleton_erl = -np.ones(num_skeleton)
    if verbose:
        erl_score.skeleton_score = [SkeletonScore() for _ in range(num_skeleton)]

    # 1. find false merges among gt skeletons
    # unique pairs of (skeleton, segment)
    skeleton_segment, count = np.unique(
        np.hstack([erl_graph.nodes[:, :1], node_segment_lut.reshape(-1, 1)]),
        axis=0,
        return_counts=True,
    )
    # AxonEM paper: only count the pairs that have intersections
    # more than merge_threshold amount of voxels
    skeleton_segment_big = skeleton_segment[count >= merge_threshold]
    # number of times that a segment was mapped to a skeleton
    segments, num_segment_skeletons = np.unique(
        skeleton_segment_big[:, 1], return_counts=True
    )
    # all segments that merge at least two skeletons
    merged_segments = segments[num_segment_skeletons > 1]
    merged_skeletons = np.unique(
        skeleton_segment[np.isin(skeleton_segment[:, 1], merged_segments), 0]
    )
    erl_score.skeleton_erl[merged_skeletons] = 0
    if verbose:
        for i in merged_skeletons:
            erl_score.skeleton_score[i].merged_seg = np.intersect1d(
                merged_segments,
                skeleton_segment_big[skeleton_segment_big[:, 0] == i, 1],
            )
            erl_score.skeleton_score[i].merged_seg_num = [0] * len(
                erl_score.skeleton_score[i].merged_seg
            )
            for j, k in enumerate(erl_score.skeleton_score[i].merged_seg):
                count_id = np.where(np.abs(skeleton_segment - [i, k]).sum(axis=1) == 0)[
                    0
                ]
                erl_score.skeleton_score[i].merged_seg_num[j] = count[count_id]
        for i in merged_segments:
            erl_score.merged_seg[i] = skeleton_segment_big[
                skeleton_segment_big[:, 1] == i, 0
            ]

    # 2. find false merges with mask
    merged_mask = None
    if mask_segment_id is not None:
        # mask for regions that are not in the same semantic class
        mask_id, mask_count = np.unique(mask_segment_id, return_counts=True)
        merged_mask = mask_id[mask_count > merge_threshold]
        merged_mask_skeletons = np.unique(
            skeleton_segment_big[np.isin(skeleton_segment_big[:, 1], merged_mask), 0]
        )
        erl_score.skeleton_erl[merged_mask_skeletons] = 0
    if verbose:
        erl_score.merged_mask = merged_mask
        for i in merged_mask_skeletons:
            erl_score.skeleton_score[i].merged_mask = np.intersect1d(
                merged_mask,
                skeleton_segment_big[skeleton_segment_big[:, 0] == i, 1],
            )
            erl_score.skeleton_score[i].merged_mask_num = [0] * len(
                erl_score.skeleton_score[i].merged_mask
            )
            for j, k in enumerate(erl_score.skeleton_score[i].merged_mask):
                count_id = np.where(np.abs(skeleton_segment - [i, k]).sum(axis=1) == 0)[
                    0
                ]
                erl_score.skeleton_score[i].merged_mask_num[j] = count[count_id]
    # 3. compute erl for correct skeletons
    for i in np.where(erl_score.skeleton_erl < 0)[0]:
        edges = erl_graph.edges[i]
        segment = node_segment_lut[edges[:, :2].astype(int)]
        # correct edges
        correct_ind = np.where(segment[:, 0] == segment[:, 1])[0]
        correct = segment[correct_ind, 0]
        correct_seg, correct_seg_ind = np.unique(correct, return_inverse=True)
        num_seg = len(correct_seg)
        correct_len = np.zeros(num_seg)
        for j in range(num_seg):
            correct_len[j] = edges[correct_ind[correct_seg_ind == j], 2].sum()
        erl_score.skeleton_erl[i] = (
            correct_len * correct_len / erl_score.skeleton_len[i]
        ).sum()
        if verbose:
            # number of edges with background nodes
            erl_score.skeleton_score[i].ommitted = (segment == 0).max(axis=1).sum()
            # number of split edges
            erl_score.skeleton_score[i].split = (segment[:, 0] != segment[:, 1]).sum()
            erl_score.skeleton_score[i].correct_seg = correct_seg
            erl_score.skeleton_score[i].correct_len = correct_len

    return erl_score
