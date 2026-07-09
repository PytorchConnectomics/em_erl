import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest


def _load_j0126_workflow():
    path = Path(__file__).resolve().parents[1] / "scripts" / "j0126_workflow.py"
    spec = importlib.util.spec_from_file_location("j0126_workflow", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeCloudVolume:
    def __init__(self, data_xyz, chunk_size):
        self.data_xyz = np.asarray(data_xyz)
        self.volume_size = np.asarray(self.data_xyz.shape, dtype=np.int64)
        self.chunk_size = np.asarray(chunk_size, dtype=np.int64)
        self.requests = []

    def __getitem__(self, item):
        assert len(item) == 3
        self.requests.append(
            tuple((axis_slice.start, axis_slice.stop) for axis_slice in item)
        )
        return self.data_xyz[item][..., np.newaxis]


class RaisingCloudVolume:
    volume_size = np.asarray([4, 4, 4], dtype=np.int64)
    chunk_size = np.asarray([4, 4, 4], dtype=np.int64)

    def __getitem__(self, item):
        raise AssertionError("CloudVolume data should not be touched")


def _tiny_skeletons():
    return {
        0: SimpleNamespace(
            vertices=np.array(
                [
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 1],
                ],
                dtype=np.float32,
            ),
            edges=np.array([[0, 1], [1, 2]], dtype=np.uint32),
        ),
        1: SimpleNamespace(
            vertices=np.array(
                [
                    [1, 0, 0],
                    [1, 0, 1],
                ],
                dtype=np.float32,
            ),
            edges=np.array([[0, 1]], dtype=np.uint32),
        ),
    }


def _write_skeleton_h5(path, skel_dict):
    with h5py.File(path, "w") as f:
        for skeleton_id, skeleton in skel_dict.items():
            group = f.create_group(str(skeleton_id))
            group.create_dataset("vertices", data=skeleton.vertices)
            group.create_dataset("edges", data=skeleton.edges)


def _fake_data_xyz():
    return np.arange(4 * 4 * 4, dtype=np.uint64).reshape(4, 4, 4) + 100


@pytest.mark.parametrize("num_workers", [1, 4])
def test_sample_cloudvolume_lut_multiple_chunks_and_oob(num_workers):
    module = _load_j0126_workflow()
    data_xyz = np.arange(8 * 8 * 8, dtype=np.uint64).reshape(8, 8, 8) + 11
    cv = FakeCloudVolume(data_xyz, chunk_size=(3, 4, 2))
    node_zyx = np.array(
        [
            [0, 0, 0],
            [1, 2, 3],
            [5, 5, 5],
            [7, 0, 7],
            [8, 0, 0],
            [-1, 0, 0],
        ],
        dtype=np.int64,
    )

    lut = module.sample_cloudvolume_lut(cv, node_zyx, num_workers=num_workers)

    expected = np.array(
        [
            data_xyz[0, 0, 0],
            data_xyz[3, 2, 1],
            data_xyz[5, 5, 5],
            data_xyz[7, 0, 7],
            0,
            0,
        ],
        dtype=np.uint64,
    )
    np.testing.assert_array_equal(lut, expected)
    assert lut.dtype == np.uint64
    assert len(cv.requests) == 4
    for request in cv.requests:
        for start, stop in request:
            assert 0 <= start < stop <= 8


def test_normalize_seg_url():
    module = _load_j0126_workflow()

    assert module.normalize_seg_url("gs://bucket/path") == "precomputed://gs://bucket/path"
    assert (
        module.normalize_seg_url("precomputed://gs://bucket/path")
        == "precomputed://gs://bucket/path"
    )
    assert (
        module.normalize_seg_url("https://example.com/layer")
        == "precomputed://https://example.com/layer"
    )
    with pytest.raises(ValueError, match="segmentation URL"):
        module.normalize_seg_url("/local/path")


def test_lut_round_trip_reuse_scores_without_cloudvolume_access(tmp_path, monkeypatch):
    module = _load_j0126_workflow()
    skel_dict = _tiny_skeletons()
    graph = module.skel_to_erlgraph(skel_dict)
    cv = FakeCloudVolume(_fake_data_xyz(), chunk_size=(2, 2, 2))

    lut = module.sample_cloudvolume_lut(
        cv,
        graph.get_nodes_position(None),
        num_workers=1,
    )
    lut_path = tmp_path / "node_segment_lut.h5"
    module.save_node_segment_lut(lut_path, lut)

    direct_score = module.score_skeletons_with_lut(
        skel_dict,
        lut,
        merge_threshold=1,
    )
    gt_path = tmp_path / "skeletons.h5"
    _write_skeleton_h5(gt_path, skel_dict)

    opened_cloudvolumes = []

    def fake_open_seg_cloudvolume(*args, **kwargs):
        opened_cloudvolumes.append((args, kwargs))
        return RaisingCloudVolume()

    monkeypatch.setattr(module, "open_seg_cloudvolume", fake_open_seg_cloudvolume)
    reuse_score = module.run_j0126_eval(
        gt_path,
        lut_path=lut_path,
        merge_threshold=1,
        num_workers=1,
    )

    assert opened_cloudvolumes == []
    loaded_lut = module.load_node_segment_lut(lut_path)
    assert loaded_lut.dtype == np.uint64
    np.testing.assert_array_equal(loaded_lut, lut)
    np.testing.assert_allclose(reuse_score.erl, direct_score.erl)
    np.testing.assert_allclose(reuse_score.skeleton_erl, direct_score.skeleton_erl)


def test_lut_length_mismatch_raises_clear_error():
    module = _load_j0126_workflow()
    graph = module.skel_to_erlgraph(_tiny_skeletons())
    bad_lut = np.zeros(graph.num_nodes + 1, dtype=np.uint64)

    with pytest.raises(
        RuntimeError,
        match="length does not match ERL graph node count",
    ):
        module.score_graph_with_lut(graph, bad_lut, lut_source="test LUT")


def test_run_j0126_eval_passes_mip_to_cloudvolume_opener(tmp_path, monkeypatch):
    module = _load_j0126_workflow()
    skel_dict = _tiny_skeletons()
    gt_path = tmp_path / "skeletons.h5"
    _write_skeleton_h5(gt_path, skel_dict)
    captured = {}

    def fake_open_seg_cloudvolume(seg_url, mip=0, cache_dir=""):
        captured["seg_url"] = seg_url
        captured["mip"] = mip
        captured["cache_dir"] = cache_dir
        return FakeCloudVolume(_fake_data_xyz(), chunk_size=(4, 4, 4))

    monkeypatch.setattr(module, "open_seg_cloudvolume", fake_open_seg_cloudvolume)
    cache_dir = str(tmp_path / "cv-cache")
    module.run_j0126_eval(
        gt_path,
        seg_url="gs://bucket/path",
        merge_threshold=1,
        num_workers=1,
        mip=2,
        cache_dir=cache_dir,
    )

    assert captured == {
        "seg_url": "gs://bucket/path",
        "mip": 2,
        "cache_dir": cache_dir,
    }


def test_open_seg_cloudvolume_passes_cache_to_constructor(monkeypatch):
    module = _load_j0126_workflow()
    captured = []

    class FakeCloudVolumeConstructor:
        def __init__(self, *args, **kwargs):
            captured.append((args, kwargs))

    monkeypatch.setitem(
        sys.modules,
        "cloudvolume",
        SimpleNamespace(CloudVolume=FakeCloudVolumeConstructor),
    )

    module.open_seg_cloudvolume("gs://bucket/path", mip=3, cache_dir="/tmp/cv-cache")
    assert captured[-1][0] == ("precomputed://gs://bucket/path",)
    assert captured[-1][1]["mip"] == 3
    assert captured[-1][1]["cache"] == "/tmp/cv-cache"

    module.open_seg_cloudvolume("https://example.com/layer")
    assert captured[-1][1]["cache"] is False
