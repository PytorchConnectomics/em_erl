import itertools
from pathlib import Path

import h5py
import numpy as np
import pytest

from em_erl.eval import compute_segment_lut, score_graph_with_lut
from em_erl.io import write_h5
from examples import eval_volume_chunk as vec


def _write_skeleton_h5(path, vertices, edges):
    with h5py.File(path, "w") as handle:
        group = handle.create_group("17")
        group.create_dataset("vertices", data=vertices)
        group.create_dataset("edges", data=edges)


def _make_chunk_fixture(tmp_path):
    factor = [3, 4, 5]
    z_range = [1, 2]
    y_range = [1, 2]
    x_range = [1, 2]
    shape = (
        (max(z_range) + 1) * factor[0],
        (max(y_range) + 1) * factor[1],
        (max(x_range) + 1) * factor[2],
    )

    zz, yy, xx = np.indices(shape, dtype=np.uint32)
    volume = (
        (zz * np.uint32(100003) + yy * np.uint32(1009) + xx * np.uint32(9176) + 37)
        % np.uint32(1000003)
    ).astype(np.uint32)
    volume += np.uint32(1000)

    vertices = np.array(
        [
            [6, 8, 10],
            [3, 4, 5],
            [8, 11, 14],
            [5, 7, 9],
            [6, 7, 9],
            [4, 8, 5],
            [5, 4, 10],
        ],
        dtype=np.float32,
    )
    edges = np.array(
        [
            [1, 3],
            [3, 4],
            [0, 2],
            [5, 0],
            [6, 1],
        ],
        dtype=np.uint32,
    )
    same_label = np.uint32(733331)
    volume[tuple(vertices[1].astype(int))] = same_label
    volume[tuple(vertices[3].astype(int))] = same_label

    seg_path_format = str(tmp_path / "seg" / "%04d" / "%d_%d.h5")
    for z, y, x in itertools.product(z_range, y_range, x_range):
        z0 = z * factor[0]
        y0 = y * factor[1]
        x0 = x * factor[2]
        chunk = volume[
            z0 : z0 + factor[0],
            y0 : y0 + factor[1],
            x0 : x0 + factor[2],
        ]
        write_h5(seg_path_format % (z, y, x), chunk)

    skeleton_path = tmp_path / "skeletons.h5"
    _write_skeleton_h5(skeleton_path, vertices, edges)

    cfg = vec.build_config(
        config_data={
            "eval_volume_chunk": {
                "seg_path_format": seg_path_format,
                "z_range": "1,3,1",
                "y_range": y_range,
                "x_range": x_range,
                "factor": factor,
                "gt_skeleton": str(skeleton_path),
                "workflow_root": str(tmp_path / "workflow"),
                "merge_threshold": 50,
                "backend": "multiprocess",
                "num_workers": 1,
            }
        }
    )
    return cfg, volume


@pytest.mark.parametrize("num_workers", [1, 2])
def test_chunked_lut_matches_monolithic_and_erl(tmp_path, num_workers):
    cfg, volume = _make_chunk_fixture(tmp_path)
    graph = vec.init_workflow(cfg)

    vec.run_chunks_local(cfg, num_workers=num_workers)
    combined_lut, chunk_score = vec.reduce_and_score(
        cfg,
        do_reduce=True,
        do_score=True,
    )
    expected_lut, _ = compute_segment_lut(volume, graph.get_nodes_position(None))
    expected_score = score_graph_with_lut(
        graph,
        expected_lut,
        merge_threshold=cfg["merge_threshold"],
    )

    assert len(combined_lut) == graph.num_nodes
    np.testing.assert_array_equal(combined_lut, expected_lut)
    np.testing.assert_allclose(chunk_score.erl, expected_score.erl)
    np.testing.assert_allclose(chunk_score.skeleton_erl, expected_score.skeleton_erl)


def test_chunk_index_key_roundtrip_and_single_chunk_output(tmp_path):
    cfg, _ = _make_chunk_fixture(tmp_path)
    vec.init_workflow(cfg)
    chunks = vec.build_chunk_list(cfg)

    for index, key in enumerate(chunks):
        assert vec.chunk_index_to_key(cfg, index) == key
        assert vec.chunk_key_to_index(cfg, key) == index

    index = 5
    key = vec.chunk_index_to_key(cfg, index)
    expected_path = Path(vec.partial_lut_path(cfg, key))
    vec.run_chunk(cfg, chunk_index=index)

    written = sorted((Path(cfg["workflow_root"]) / "lut").glob("*.h5"))
    assert written == [expected_path]


def test_wait_complete_and_stall_reports_missing_chunk(tmp_path):
    cfg, _ = _make_chunk_fixture(tmp_path)
    expected_paths = vec.expected_partial_lut_paths(cfg)
    for path in expected_paths:
        write_h5(path, np.zeros(1, dtype=np.uint8))

    assert vec.wait_for_completion(cfg, stall_timeout=0.05, poll_interval=0.01)

    missing_key = vec.chunk_index_to_key(cfg, 3)
    missing_path = vec.partial_lut_path(cfg, missing_key)
    Path(missing_path).unlink()

    with pytest.raises(vec.WaitTimeoutError) as exc:
        vec.wait_for_completion(cfg, stall_timeout=0.03, poll_interval=0.01)

    message = str(exc.value)
    assert "chunk 3" in message
    assert missing_path in message


def test_reduce_length_matches_num_nodes(tmp_path):
    cfg, _ = _make_chunk_fixture(tmp_path)
    graph = vec.init_workflow(cfg)
    vec.run_chunks_local(cfg, num_workers=1)

    combined_lut, _ = vec.reduce_and_score(cfg, do_reduce=True, do_score=False)

    assert len(combined_lut) == graph.num_nodes


def test_build_sbatch_script_contains_array_and_worker_command(tmp_path):
    cfg, _ = _make_chunk_fixture(tmp_path)
    config_path = tmp_path / "eval_volume_chunk.yaml"
    script = vec.build_sbatch_script(cfg, config_path=config_path)
    n_chunks = len(vec.build_chunk_list(cfg))

    assert f"--array=0-{n_chunks - 1}" in script
    assert "--chunk-index $SLURM_ARRAY_TASK_ID" in script


def test_config_block_and_overrides_are_normalized(tmp_path):
    cfg, _ = _make_chunk_fixture(tmp_path)
    updated = vec.build_config(
        config_data={"eval_volume_chunk": cfg},
        overrides=["merge_threshold=25", "slurm.mem=4G", "backend=slurm"],
    )

    assert updated["z_range"] == [1, 2]
    assert updated["factor"] == [3, 4, 5]
    assert updated["merge_threshold"] == 25
    assert updated["backend"] == "slurm"
    assert updated["slurm"]["mem"] == "4G"
