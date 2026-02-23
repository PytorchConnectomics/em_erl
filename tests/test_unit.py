import tempfile
import os
import numpy as np
import pytest

from em_erl.erl import ERLGraph, ERLScore, SkeletonScore, skel_to_erlgraph
from em_erl.eval import compute_segment_lut, compute_erl_score, compute_segment_lut_tile
from em_erl.io import read_h5, write_h5, read_pkl, write_pkl
from em_erl.skel import cable_length


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
        g = ERLGraph()
        g.skeleton_id = np.array([10, 20])
        g.skeleton_len = np.array([100.0, 200.0])
        g.nodes = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 10],
            [1, 5, 5, 5],
            [1, 5, 5, 15],
        ], dtype=np.uint32)
        g.edges = [
            np.array([[0, 1, 10.0]]),
            np.array([[2, 3, 10.0]]),
        ]
        path = str(tmp_path / "test_graph.npz")
        g.save_npz(path)

        g2 = ERLGraph(path)
        np.testing.assert_array_equal(g.skeleton_id, g2.skeleton_id)
        np.testing.assert_array_equal(g.skeleton_len, g2.skeleton_len)
        np.testing.assert_array_equal(g.nodes, g2.nodes)
        assert len(g.edges) == len(g2.edges)
        for e1, e2 in zip(g.edges, g2.edges):
            np.testing.assert_allclose(e1, e2)

    def test_get_nodes_position(self):
        g = ERLGraph()
        g.nodes = np.array([
            [0, 30, 60, 90],
            [1, 60, 120, 180],
        ], dtype=np.uint32)
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


class TestComputeErlScore:
    def test_perfect_segmentation(self):
        """All nodes map to the same segment per skeleton → ERL = skeleton length."""
        g = ERLGraph()
        g.skeleton_id = np.array([0])
        g.skeleton_len = np.array([100.0])
        g.nodes = np.array([[0, 0, 0, 0], [0, 0, 0, 1]], dtype=np.uint32)
        g.edges = [np.array([[0, 1, 100.0]])]

        node_lut = np.array([5, 5], dtype=np.uint32)  # same segment
        score = compute_erl_score(g, node_lut, None, merge_threshold=0)
        assert score.skeleton_erl[0] == pytest.approx(100.0)

    def test_split_segmentation(self):
        """Nodes map to different segments → ERL < skeleton length."""
        g = ERLGraph()
        g.skeleton_id = np.array([0])
        g.skeleton_len = np.array([100.0])
        g.nodes = np.array([[0, 0, 0, 0], [0, 0, 0, 1]], dtype=np.uint32)
        g.edges = [np.array([[0, 1, 100.0]])]

        node_lut = np.array([5, 6], dtype=np.uint32)  # different segments
        score = compute_erl_score(g, node_lut, None, merge_threshold=0)
        assert score.skeleton_erl[0] == pytest.approx(0.0)


class TestH5Roundtrip:
    def test_single_dataset(self, tmp_path):
        data = np.array([1, 2, 3], dtype=np.int32)
        path = str(tmp_path / "test.h5")
        write_h5(path, data)
        loaded = read_h5(path)
        np.testing.assert_array_equal(data, loaded)

    def test_multiple_datasets(self, tmp_path):
        d1 = np.array([1.0, 2.0])
        d2 = np.array([3.0, 4.0])
        path = str(tmp_path / "test.h5")
        write_h5(path, [d1, d2], ["a", "b"])
        r1, r2 = read_h5(path)
        np.testing.assert_array_equal(d1, r1)
        np.testing.assert_array_equal(d2, r2)


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
        gt_graph = ERLGraph(gt_path)
        gt_mask = read_vol(mask_path)

        resolution = np.array([30, 30, 30])
        node_position = gt_graph.get_nodes_position(resolution)

        node_lut, mask_id = compute_segment_lut(pred_seg, node_position, gt_mask)
        score = compute_erl_score(gt_graph, node_lut, mask_id, merge_threshold=0)
        score.compute_erl()

        assert score.erl[0] > 0  # ERL should be positive
        assert score.erl[1] > 0  # gt ERL should be positive
        assert score.erl[2] == 2  # 2 skeletons
