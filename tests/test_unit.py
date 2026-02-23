import tempfile
import os
import numpy as np
import pytest

from em_erl.erl import ERLGraph, ERLScore, SkeletonScore, skel_to_erlgraph
from em_erl.eval import (
    compute_segment_lut,
    compute_erl_score,
    compute_segment_lut_tile,
    compute_segment_lut_tile_zyx,
    combine_segment_lut_tile_zyx,
)
from em_erl.io import read_h5, write_h5, read_pkl, write_pkl
from em_erl.skel import cable_length


def make_graph(
    skeleton_id,
    skeleton_len,
    node_skeleton_index,
    node_coords_zyx,
    edge_u,
    edge_v,
    edge_len,
    edge_ptr,
):
    return ERLGraph(
        skeleton_id=np.asarray(skeleton_id),
        skeleton_len=np.asarray(skeleton_len, dtype=float),
        node_skeleton_index=np.asarray(node_skeleton_index, dtype=np.uint32),
        node_coords_zyx=np.asarray(node_coords_zyx),
        edge_u=np.asarray(edge_u, dtype=np.uint32),
        edge_v=np.asarray(edge_v, dtype=np.uint32),
        edge_len=np.asarray(edge_len, dtype=np.float32),
        edge_ptr=np.asarray(edge_ptr, dtype=np.uint64),
    )


class TestSkeletonScore:
    def test_defaults(self):
        s = SkeletonScore()
        assert s.omitted == 0
        assert s.split == 0
        assert s.merged_seg == []
        assert s.correct_seg == []


class TestERLScore:
    def test_compute_erl(self):
        score = ERLScore(
            skeleton_id=np.array([0, 1]),
            skeleton_len=np.array([100.0, 200.0]),
            verbose=False,
        )
        score.skeleton_erl = np.array([80.0, 150.0])
        score.compute_erl()
        assert score.erl is not None
        assert score.erl.shape == (3,)
        assert score.erl[2] == 2  # num_skel

    def test_compute_erl_with_intervals(self):
        score = ERLScore(
            skeleton_id=np.array([0, 1]),
            skeleton_len=np.array([50.0, 200.0]),
            verbose=False,
        )
        score.skeleton_erl = np.array([40.0, 180.0])
        score.compute_erl(erl_intervals=[0, 100, 300])
        assert score.erl.shape == (3, 3)


class TestERLGraph:
    def test_save_load_roundtrip(self, tmp_path):
        g = make_graph(
            skeleton_id=[10, 20],
            skeleton_len=[100.0, 200.0],
            node_skeleton_index=[0, 0, 1, 1],
            node_coords_zyx=[
                [0, 0, 0],
                [0, 0, 10],
                [5, 5, 5],
                [5, 5, 15],
            ],
            edge_u=[0, 2],
            edge_v=[1, 3],
            edge_len=[10.0, 10.0],
            edge_ptr=[0, 1, 2],
        )
        path = str(tmp_path / "test_graph.npz")
        g.save_npz(path)

        g2 = ERLGraph.from_npz(path)
        np.testing.assert_array_equal(g.skeleton_id, g2.skeleton_id)
        np.testing.assert_array_equal(g.skeleton_len, g2.skeleton_len)
        np.testing.assert_array_equal(g.node_skeleton_index, g2.node_skeleton_index)
        np.testing.assert_array_equal(g.node_coords_zyx, g2.node_coords_zyx)
        np.testing.assert_array_equal(g.edge_u, g2.edge_u)
        np.testing.assert_array_equal(g.edge_v, g2.edge_v)
        np.testing.assert_allclose(g.edge_len, g2.edge_len)
        np.testing.assert_array_equal(g.edge_ptr, g2.edge_ptr)

    def test_get_nodes_position(self):
        g = make_graph(
            skeleton_id=[1, 2],
            skeleton_len=[1.0, 1.0],
            node_skeleton_index=[0, 1],
            node_coords_zyx=[[30, 60, 90], [60, 120, 180]],
            edge_u=[],
            edge_v=[],
            edge_len=[],
            edge_ptr=[0, 0, 0],
        )
        pos = g.get_nodes_position(resolution=np.array([30, 30, 30]))
        np.testing.assert_array_equal(pos, [[1, 2, 3], [2, 4, 6]])


class TestCableLength:
    def test_simple(self):
        vertices = np.array([[0, 0, 0], [3, 4, 0]], dtype=float)
        edges = np.array([[0, 1]])
        assert cable_length(vertices, edges) == pytest.approx(5.0)

    def test_with_resolution(self):
        vertices = np.array([[0, 0, 0], [1, 1, 0]], dtype=float)
        edges = np.array([[0, 1]])
        length = cable_length(vertices, edges, res=[2, 2, 2])
        assert length == pytest.approx(2 * np.sqrt(2))

    def test_empty_edges(self):
        vertices = np.array([[0, 0, 0]])
        assert cable_length(vertices, np.array([]).reshape(0, 2)) == 0


class TestComputeSegmentLutTile:
    def test_returns_ind_and_val(self):
        seg = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.uint32)
        pts = np.array([[0, 0, 0], [0, 0, 1], [1, 1, 1]], dtype=int)
        ind, val = compute_segment_lut_tile(seg, pts)
        assert ind.shape == (3,)
        assert val.shape[0] > 0


class TestComputeSegmentLut:
    def test_chunked_h5_matches_array(self, tmp_path):
        seg = np.arange(4 * 5 * 6, dtype=np.uint32).reshape(4, 5, 6)
        mask = np.zeros_like(seg, dtype=np.uint8)
        mask[0, 1, 2] = 1
        mask[2, 4, 5] = 1
        mask[3, 0, 0] = 1

        seg_path = str(tmp_path / "seg.h5")
        mask_path = str(tmp_path / "mask.h5")
        write_h5(seg_path, seg)
        write_h5(mask_path, mask)

        pts = np.array(
            [
                [0, 1, 2],
                [1, 3, 4],
                [2, 4, 5],
                [3, 0, 0],
            ],
            dtype=int,
        )

        lut_arr, mask_ids_arr = compute_segment_lut(seg, pts, mask=mask, chunk_num=1)
        lut_h5, mask_ids_h5 = compute_segment_lut(seg_path, pts, mask=mask_path, chunk_num=3)

        np.testing.assert_array_equal(lut_arr, lut_h5)
        np.testing.assert_array_equal(np.sort(mask_ids_arr), np.sort(mask_ids_h5))


class TestComputeSegmentLutTileZyx:
    def test_skips_reading_empty_tiles(self, tmp_path):
        seg_path_format = str(tmp_path / "seg" / "%04d" / "%d_%d.h5")
        out_path_format = str(tmp_path / "lut" / "%04d" / "%d_%d.h5")

        os.makedirs(os.path.dirname(seg_path_format % (0, 0, 0)), exist_ok=True)
        seg = np.zeros((2, 10, 10), dtype=np.uint32)
        seg[0, 1, 1] = 11
        seg[1, 2, 3] = 22
        write_h5(seg_path_format % (0, 0, 0), seg)
        # Intentionally do not create (0,0,1), (0,1,0), (0,1,1)

        pts = np.array([[0, 1, 1], [1, 2, 3]], dtype=int)
        compute_segment_lut_tile_zyx(
            seg_path_format=seg_path_format,
            zran=[0],
            yran=[0, 1],
            xran=[0, 1],
            pts=pts,
            output_path_format=out_path_format,
            factor=[1, 10, 10],
        )

        for y in [0, 1]:
            for x in [0, 1]:
                assert os.path.exists(out_path_format % (0, y, x))

        out = combine_segment_lut_tile_zyx([0], [0, 1], [0, 1], out_path_format)
        np.testing.assert_array_equal(out, np.array([11, 22], dtype=np.uint32))


class TestComputeErlScore:
    def test_perfect_segmentation(self):
        """All nodes map to the same segment per skeleton → ERL = skeleton length."""
        g = make_graph(
            skeleton_id=[0],
            skeleton_len=[100.0],
            node_skeleton_index=[0, 0],
            node_coords_zyx=[[0, 0, 0], [0, 0, 1]],
            edge_u=[0],
            edge_v=[1],
            edge_len=[100.0],
            edge_ptr=[0, 1],
        )

        node_lut = np.array([5, 5], dtype=np.uint32)  # same segment
        score = compute_erl_score(g, node_lut, None, merge_threshold=0)
        assert score.skeleton_erl[0] == pytest.approx(100.0)

    def test_split_segmentation(self):
        """Nodes map to different segments → ERL < skeleton length."""
        g = make_graph(
            skeleton_id=[0],
            skeleton_len=[100.0],
            node_skeleton_index=[0, 0],
            node_coords_zyx=[[0, 0, 0], [0, 0, 1]],
            edge_u=[0],
            edge_v=[1],
            edge_len=[100.0],
            edge_ptr=[0, 1],
        )

        node_lut = np.array([5, 6], dtype=np.uint32)  # different segments
        score = compute_erl_score(g, node_lut, None, merge_threshold=0)
        assert score.skeleton_erl[0] == pytest.approx(0.0)

    def test_grouped_correct_lengths(self):
        g = make_graph(
            skeleton_id=[0],
            skeleton_len=[60.0],
            node_skeleton_index=[0, 0, 0, 0, 0, 0],
            node_coords_zyx=[
                [0, 0, 0],
                [0, 0, 1],
                [0, 0, 2],
                [0, 0, 3],
                [0, 0, 4],
                [0, 0, 5],
            ],
            edge_u=[0, 2, 4],
            edge_v=[1, 3, 5],
            edge_len=[10.0, 20.0, 30.0],
            edge_ptr=[0, 3],
        )
        # first two edges in seg 5 (sum=30), last edge in seg 9 (sum=30)
        node_lut = np.array([5, 5, 5, 5, 9, 9], dtype=np.uint32)

        score = compute_erl_score(g, node_lut, None, merge_threshold=0)
        assert score.skeleton_erl[0] == pytest.approx((30.0**2 + 30.0**2) / 60.0)

    def test_verbose_without_mask(self):
        g = make_graph(
            skeleton_id=[0],
            skeleton_len=[30.0],
            node_skeleton_index=[0, 0, 0],
            node_coords_zyx=[[0, 0, 0], [0, 0, 1], [0, 0, 2]],
            edge_u=[0, 1],
            edge_v=[1, 2],
            edge_len=[10.0, 20.0],
            edge_ptr=[0, 2],
        )
        node_lut = np.array([5, 0, 0], dtype=np.uint32)

        score = compute_erl_score(g, node_lut, None, merge_threshold=1, verbose=True)
        assert score.merged_mask is None
        assert score.skeleton_score[0].omitted == 2
        assert score.skeleton_score[0].split == 1
        np.testing.assert_array_equal(score.skeleton_score[0].correct_seg, np.array([0], dtype=np.uint32))
        np.testing.assert_allclose(score.skeleton_score[0].correct_len, np.array([20.0]))


class TestH5Roundtrip:
    def test_single_dataset(self, tmp_path):
        data = np.array([1, 2, 3], dtype=np.int32)
        path = str(tmp_path / "test.h5")
        write_h5(path, data)
        loaded = read_h5(path)
        np.testing.assert_array_equal(data, loaded)

    def test_single_dataset_by_name(self, tmp_path):
        data = np.array([4, 5, 6], dtype=np.int32)
        path = str(tmp_path / "named.h5")
        write_h5(path, data, dataset="main")
        loaded = read_h5(path, dataset="main")
        np.testing.assert_array_equal(data, loaded)

    def test_multiple_datasets(self, tmp_path):
        d1 = np.array([1.0, 2.0])
        d2 = np.array([3.0, 4.0])
        path = str(tmp_path / "test.h5")
        write_h5(path, [d1, d2], ["a", "b"])
        r1, r2 = read_h5(path)
        np.testing.assert_array_equal(d1, r1)
        np.testing.assert_array_equal(d2, r2)

    def test_multiple_datasets_explicit_list(self, tmp_path):
        d1 = np.array([7, 8], dtype=np.int32)
        d2 = np.array([9, 10], dtype=np.int32)
        path = str(tmp_path / "multi_named.h5")
        write_h5(path, [d1, d2], ["x", "y"])
        out = read_h5(path, dataset=["x", "y"])
        assert isinstance(out, list)
        np.testing.assert_array_equal(out[0], d1)
        np.testing.assert_array_equal(out[1], d2)


class TestPklRoundtrip:
    def test_single_object(self, tmp_path):
        path = str(tmp_path / "test.pkl")
        write_pkl(path, {"key": "value"})
        loaded = read_pkl(path)
        assert loaded == {"key": "value"}

    def test_list_of_objects(self, tmp_path):
        path = str(tmp_path / "test.pkl")
        write_pkl(path, [1, 2, 3])
        loaded = read_pkl(path)
        assert loaded == [1, 2, 3]


class TestEndToEnd:
    """End-to-end test using the shipped test data."""

    def test_volume_evaluation(self):
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        pred_path = os.path.join(data_dir, "vol_pred.h5")
        gt_path = os.path.join(data_dir, "gt_graph.npz")
        mask_path = os.path.join(data_dir, "vol_no-mask.h5")

        if not os.path.exists(gt_path):
            pytest.skip("gt_graph.npz not found")

        from em_erl.io import read_vol

        pred_seg = read_vol(pred_path)
        gt_graph = ERLGraph.from_npz(gt_path)
        gt_mask = read_vol(mask_path)

        resolution = np.array([30, 30, 30])
        node_position = gt_graph.get_nodes_position(resolution)

        node_lut, mask_id = compute_segment_lut(pred_seg, node_position, gt_mask)
        score = compute_erl_score(gt_graph, node_lut, mask_id, merge_threshold=0)
        score.compute_erl()

        assert score.erl[0] > 0  # ERL should be positive
        assert score.erl[1] > 0  # gt ERL should be positive
        assert score.erl[2] == 2  # 2 skeletons
