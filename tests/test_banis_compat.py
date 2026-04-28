import importlib.util
import sys
import types
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from em_erl.erl import ERLGraph
from em_erl.eval import compute_erl_score


def _load_banis_metrics(monkeypatch):
    """Load BANIS metrics against the local pure-Python funlib run_length module."""

    funlib_root = Path("/projects/weilab/weidf/lib/others/funlib.evaluate")
    run_length_path = funlib_root / "funlib" / "evaluate" / "run_length.py"
    spec = importlib.util.spec_from_file_location(
        "funlib.evaluate.run_length",
        run_length_path,
    )
    run_length = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(run_length)

    funlib_mod = types.ModuleType("funlib")
    evaluate_mod = types.ModuleType("funlib.evaluate")
    numpy_groupies_mod = types.ModuleType("numpy_groupies")

    evaluate_mod.__path__ = [str(funlib_root / "funlib" / "evaluate")]
    evaluate_mod.rand_voi = lambda *args, **kwargs: {}
    evaluate_mod.expected_run_length = run_length.expected_run_length
    evaluate_mod.get_skeleton_lengths = run_length.get_skeleton_lengths
    numpy_groupies_mod.aggregate = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "funlib", funlib_mod)
    monkeypatch.setitem(sys.modules, "funlib.evaluate", evaluate_mod)
    monkeypatch.setitem(sys.modules, "funlib.evaluate.run_length", run_length)
    monkeypatch.setitem(sys.modules, "numpy_groupies", numpy_groupies_mod)

    metrics_path = Path(__file__).resolve().parents[2] / "banis" / "metrics.py"
    spec = importlib.util.spec_from_file_location("banis_metrics_for_erl_test", metrics_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_matching_graphs():
    skeleton_ids = np.array([10, 20, 30], dtype=np.uint64)
    skeleton_lens = np.array([50.0, 20.0, 30.0], dtype=np.float64)
    node_skeleton_index = np.array([0, 0, 0, 0, 1, 1, 2, 2], dtype=np.uint32)
    node_coords_zyx = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 2],
            [0, 0, 3],
            [5, 5, 5],
            [5, 5, 6],
            [9, 9, 9],
            [9, 9, 10],
        ],
        dtype=np.float32,
    )
    edge_u = np.array([0, 1, 2, 4, 6], dtype=np.uint32)
    edge_v = np.array([1, 2, 3, 5, 7], dtype=np.uint32)
    edge_len = np.array([10.0, 15.0, 25.0, 20.0, 30.0], dtype=np.float32)
    edge_ptr = np.array([0, 3, 4, 5], dtype=np.uint64)

    erl_graph = ERLGraph(
        skeleton_id=skeleton_ids,
        skeleton_len=skeleton_lens,
        node_skeleton_index=node_skeleton_index,
        node_coords_zyx=node_coords_zyx,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_len=edge_len,
        edge_ptr=edge_ptr,
    )

    nx_graph = nx.Graph()
    node_to_skeleton_id = skeleton_ids[node_skeleton_index]
    for node_id, skeleton_id in enumerate(node_to_skeleton_id):
        nx_graph.add_node(int(node_id), id=int(skeleton_id))
    for u, v, length in zip(edge_u, edge_v, edge_len):
        nx_graph.add_edge(int(u), int(v), edge_length=float(length))

    skeleton_lengths = {
        int(skeleton_id): float(length)
        for skeleton_id, length in zip(skeleton_ids, skeleton_lens)
    }
    return erl_graph, nx_graph, skeleton_lengths


def _em_erl_nerl(erl_graph, node_segment_lut):
    score = compute_erl_score(erl_graph, node_segment_lut, None, merge_threshold=1)
    score.compute_erl()

    gt_lut = erl_graph.skeleton_id[erl_graph.node_skeleton_index].astype(np.uint32)
    max_score = compute_erl_score(erl_graph, gt_lut, None, merge_threshold=1)
    max_score.compute_erl()
    return score.erl[0] / max_score.erl[0], score.erl[0], max_score.erl[0]


def _banis_nerl(banis_metrics, nx_graph, skeleton_lengths, node_segment_lut):
    erl = banis_metrics.adapted_erl(
        nx_graph,
        "id",
        "edge_length",
        node_segment_lut,
        skeleton_lengths=skeleton_lengths,
        ignored_merger_size=0,
    )
    max_erl = banis_metrics.adapted_erl(
        nx_graph,
        "id",
        "edge_length",
        {node: data["id"] for node, data in nx_graph.nodes(data=True)},
        skeleton_lengths=skeleton_lengths,
        ignored_merger_size=0,
    )
    return erl / max_erl, erl, max_erl


def test_em_erl_nerl_matches_banis_adapted_erl(monkeypatch):
    banis_metrics = _load_banis_metrics(monkeypatch)
    erl_graph, nx_graph, skeleton_lengths = _make_matching_graphs()

    # Segment 100 merges skeletons 10 and 20. Skeleton 30 is perfectly segmented.
    em_lut = np.array([7, 7, 100, 100, 100, 100, 3, 3], dtype=np.uint32)
    banis_lut = {node: int(seg) for node, seg in enumerate(em_lut)}

    em_nerl, em_erl, em_max_erl = _em_erl_nerl(erl_graph, em_lut)
    banis_nerl, banis_erl, banis_max_erl = _banis_nerl(
        banis_metrics,
        nx_graph,
        skeleton_lengths,
        banis_lut,
    )

    assert em_erl == pytest.approx(banis_erl)
    assert em_max_erl == pytest.approx(banis_max_erl)
    assert em_nerl == pytest.approx(banis_nerl)
