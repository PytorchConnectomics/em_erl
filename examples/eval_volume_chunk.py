import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import copy
import itertools
import multiprocessing
import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import time

from em_erl.io import load_skeletons, read_vol, write_h5
from em_erl.erl import ERLGraph, skel_to_erlgraph
from em_erl.eval import (
    combine_segment_lut_tile_zyx,
    compute_segment_lut_tile_zyx,
    score_graph_with_lut,
)


DEFAULT_CONDA_INIT = "/projects/weilab/weidf/lib/miniconda3/bin/activate"
DEFAULT_CONDA_ENV = "pytc"


class WaitTimeoutError(RuntimeError):
    def __init__(self, message, missing_chunks):
        super().__init__(message)
        self.missing_chunks = missing_chunks


def _coerce_override_value(value):
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            pass
    return value


def apply_overrides(config, overrides):
    config = copy.deepcopy(config)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Invalid override {override!r}; expected key=value")
        key, value = override.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid override {override!r}; empty key")
        target = config
        parts = key.split(".")
        for part in parts[:-1]:
            child = target.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError(f"Override parent {part!r} is not a mapping")
            target = child
        target[parts[-1]] = _coerce_override_value(value.strip())
    return config


def parse_range(spec):
    if isinstance(spec, str):
        values = [x.strip() for x in spec.split(",") if x.strip() != ""]
        if len(values) != 3:
            raise ValueError(
                f"Range string {spec!r} must be 'start,stop,step'"
            )
        start, stop, step = [int(x) for x in values]
        return list(range(start, stop, step))
    if isinstance(spec, int):
        return [int(spec)]
    try:
        return [int(x) for x in spec]
    except TypeError as exc:
        raise ValueError(f"Range must be a list or 'start,stop,step', got {spec!r}") from exc


def parse_factor(spec):
    if spec is None:
        factor = [1, 1, 1]
    elif isinstance(spec, str):
        factor = [int(x.strip()) for x in spec.split(",") if x.strip() != ""]
    else:
        factor = [int(x) for x in spec]
    if len(factor) != 3:
        raise ValueError(f"factor must have length 3, got {factor!r}")
    return factor


def normalize_config(config):
    cfg = dict(config)
    required = ["seg_path_format", "workflow_root", "z_range", "y_range", "x_range"]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")

    cfg["workflow_root"] = str(Path(cfg["workflow_root"]).expanduser())
    cfg["z_range"] = parse_range(cfg["z_range"])
    cfg["y_range"] = parse_range(cfg["y_range"])
    cfg["x_range"] = parse_range(cfg["x_range"])
    cfg["factor"] = parse_factor(cfg.get("factor", [1, 1, 1]))
    cfg["seg_oset"] = int(cfg.get("seg_oset", 0))
    cfg["merge_threshold"] = int(cfg.get("merge_threshold", 50))
    cfg["num_workers"] = int(cfg.get("num_workers", 1))
    cfg["backend"] = cfg.get("backend", "multiprocess")
    cfg["dataset"] = cfg.get("dataset", None) or None
    cfg["output_path"] = cfg.get("output_path", "")
    cfg["slurm"] = dict(cfg.get("slurm") or {})
    return cfg


def build_config(config_path=None, overrides=None, config_data=None):
    if config_data is None:
        if config_path is None:
            raise ValueError("config_path is required")
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required to load --config YAML files") from exc
        with open(config_path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        block = raw.get("eval_volume_chunk", {})
    else:
        block = config_data.get("eval_volume_chunk", config_data)

    cfg = apply_overrides(block, overrides or [])
    if config_path is not None:
        cfg["_config_path"] = str(Path(config_path).expanduser().resolve())
    return normalize_config(cfg)


def build_chunk_list(config):
    cfg = normalize_config(config)
    return list(itertools.product(cfg["z_range"], cfg["y_range"], cfg["x_range"]))


def _as_chunk_list(config_or_chunks):
    if isinstance(config_or_chunks, dict):
        return build_chunk_list(config_or_chunks)
    return list(config_or_chunks)


def chunk_index_to_key(config_or_chunks, index):
    chunks = _as_chunk_list(config_or_chunks)
    index = int(index)
    if index < 0 or index >= len(chunks):
        raise IndexError(f"chunk index {index} out of range 0..{len(chunks) - 1}")
    return chunks[index]


def chunk_key_to_index(config_or_chunks, key):
    chunks = _as_chunk_list(config_or_chunks)
    key = tuple(int(x) for x in key)
    try:
        return chunks.index(key)
    except ValueError as exc:
        raise ValueError(f"chunk key {key!r} is not in the configured grid") from exc


def vertices_path(config):
    return str(Path(config["workflow_root"]) / "gt_vertices.h5")


def graph_path(config):
    return str(Path(config["workflow_root"]) / "gt_graph.npz")


def lut_output_path_format(config):
    return str(Path(config["workflow_root"]) / "lut" / "%04d_%d_%d.h5")


def combined_lut_path(config):
    return str(Path(config["workflow_root"]) / "seg_lut_all.h5")


def partial_lut_path(config, key):
    return lut_output_path_format(config) % tuple(key)


def expected_partial_lut_paths(config):
    cfg = normalize_config(config)
    return [partial_lut_path(cfg, key) for key in build_chunk_list(cfg)]


def print_workflow_summary(config, graph=None):
    cfg = normalize_config(config)
    chunks = build_chunk_list(cfg)
    print(f"workflow_root: {cfg['workflow_root']}")
    print(
        "grid: "
        f"z={len(cfg['z_range'])}, y={len(cfg['y_range'])}, "
        f"x={len(cfg['x_range'])}, chunks={len(chunks)}"
    )
    print(f"factor: {cfg['factor']}")
    print(f"backend: {cfg['backend']}")
    if graph is not None:
        print(
            "graph: "
            f"skeletons={graph.num_skeletons}, nodes={graph.num_nodes}, "
            f"edges={graph.num_edges}"
        )
    print(f"vertices: {vertices_path(cfg)}")
    print(f"graph_npz: {graph_path(cfg)}")
    print(f"partial_lut_format: {lut_output_path_format(cfg)}")
    print(f"combined_lut: {combined_lut_path(cfg)}")


def init_workflow(config):
    cfg = normalize_config(config)
    Path(cfg["workflow_root"]).mkdir(parents=True, exist_ok=True)

    if cfg.get("gt_graph"):
        graph = ERLGraph.from_npz(cfg["gt_graph"])
    elif cfg.get("gt_skeleton"):
        skeletons = load_skeletons(cfg["gt_skeleton"])
        graph = skel_to_erlgraph(skeletons)
    else:
        raise ValueError("Config must provide gt_skeleton or gt_graph for --init-only")

    pts = graph.get_nodes_position(None)
    write_h5(vertices_path(cfg), pts)
    graph.save_npz(graph_path(cfg))
    print("initialized chunked volume evaluation workflow")
    print_workflow_summary(cfg, graph)
    return graph


def _chunk_indices_from_range(chunk_range):
    out = []
    if not chunk_range:
        return out
    for part in chunk_range.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid chunk range {part!r}: end < start")
            out.extend(range(start, end + 1))
        else:
            out.append(int(part))
    return out


def run_chunk(config, chunk_index=None, chunk_key=None):
    cfg = normalize_config(config)
    if chunk_key is None:
        if chunk_index is None:
            env_index = os.environ.get("SLURM_ARRAY_TASK_ID")
            if env_index is None:
                raise ValueError("chunk_index is required outside a SLURM array task")
            chunk_index = int(env_index)
        chunk_key = chunk_index_to_key(cfg, chunk_index)
    else:
        chunk_key = tuple(int(x) for x in chunk_key)
        chunk_index = chunk_key_to_index(cfg, chunk_key)

    z, y, x = chunk_key
    pts = read_vol(vertices_path(cfg))
    compute_segment_lut_tile_zyx(
        cfg["seg_path_format"],
        [z],
        [y],
        [x],
        pts,
        lut_output_path_format(cfg),
        factor=cfg["factor"],
        dataset=cfg["dataset"],
        seg_oset=cfg["seg_oset"],
    )
    output_path = partial_lut_path(cfg, chunk_key)
    print(f"chunk {chunk_index} key={chunk_key} output={output_path}")
    return output_path


def run_chunk_indices(config, chunk_indices):
    return [run_chunk(config, chunk_index=index) for index in chunk_indices]


def _set_worker_thread_env():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"


def _run_chunk_subset_worker(payload):
    config, indices = payload
    _set_worker_thread_env()
    return run_chunk_indices(config, indices)


def run_chunks_local(config, chunk_indices=None, num_workers=1):
    cfg = normalize_config(config)
    chunks = build_chunk_list(cfg)
    if chunk_indices is None:
        chunk_indices = list(range(len(chunks)))
    else:
        chunk_indices = [int(x) for x in chunk_indices]
    if len(chunk_indices) == 0:
        return []

    num_workers = max(1, int(num_workers))
    print(f"mapping {len(chunk_indices)} chunks with {num_workers} local worker(s)")
    if num_workers == 1:
        return run_chunk_indices(cfg, chunk_indices)

    subsets = [
        chunk_indices[i::num_workers]
        for i in range(num_workers)
        if len(chunk_indices[i::num_workers]) > 0
    ]
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=len(subsets)) as pool:
        nested = pool.map(_run_chunk_subset_worker, [(cfg, subset) for subset in subsets])
    return [path for group in nested for path in group]


def _slurm_value(slurm, *keys, default=""):
    for key in keys:
        if key in slurm:
            return slurm[key]
    return default


def build_sbatch_script(config, config_path=None):
    cfg = normalize_config(config)
    n_chunks = len(build_chunk_list(cfg))
    if n_chunks <= 0:
        raise ValueError("Cannot build sbatch script for an empty chunk grid")
    slurm = cfg.get("slurm", {})
    abs_config = Path(config_path or cfg.get("_config_path", "")).expanduser()
    if str(abs_config) == ".":
        raise ValueError("config_path is required to build an sbatch script")
    abs_config = abs_config.resolve()
    repo_root = Path(__file__).resolve().parents[1]
    slurm_dir = Path(cfg["workflow_root"]) / "slurm_outputs"
    conda_init = _slurm_value(slurm, "conda_init", default=DEFAULT_CONDA_INIT)
    conda_env = _slurm_value(slurm, "conda_env", default=DEFAULT_CONDA_ENV)

    return textwrap.dedent(
        f"""\
        #!/bin/bash
        #SBATCH --job-name=eval_volume_chunk
        #SBATCH --array=0-{n_chunks - 1}
        #SBATCH --partition={_slurm_value(slurm, "partition", default="default")}
        #SBATCH --mem={_slurm_value(slurm, "mem", default="16G")}
        #SBATCH --cpus-per-task={_slurm_value(slurm, "cpus_per_task", "cpus-per-task", default=1)}
        #SBATCH --time={_slurm_value(slurm, "time", default="12:00:00")}
        #SBATCH --output={slurm_dir}/%A_%a.out
        #SBATCH --error={slurm_dir}/%A_%a.err

        set -euo pipefail
        cd {repo_root}
        source {conda_init} {conda_env}
        python -u examples/eval_volume_chunk.py --config {abs_config} --chunk-index $SLURM_ARRAY_TASK_ID
        """
    )


def submit_sbatch(config, config_path=None):
    cfg = normalize_config(config)
    slurm_dir = Path(cfg["workflow_root"]) / "slurm_outputs"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    script = build_sbatch_script(cfg, config_path=config_path)
    with tempfile.NamedTemporaryFile("w", suffix=".sbatch", delete=False) as handle:
        handle.write(script)
        script_path = handle.name

    result = subprocess.run(
        ["sbatch", script_path],
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip(), file=sys.stderr)
    monitor_config = Path(config_path or cfg.get("_config_path", "")).expanduser().resolve()
    print(
        "monitor: "
        f"python examples/eval_volume_chunk.py --config {monitor_config} --wait"
    )
    return result


def _missing_chunks(config, chunks=None):
    cfg = normalize_config(config)
    if chunks is None:
        chunks = build_chunk_list(cfg)
    missing = []
    for index, key in enumerate(chunks):
        path = partial_lut_path(cfg, key)
        if not os.path.exists(path):
            missing.append((index, key, path))
    return missing


def _format_missing_message(prefix, missing, total):
    lines = [f"{prefix}; missing {len(missing)}/{total} chunks:"]
    for index, key, path in missing[:20]:
        lines.append(f"  chunk {index} key={key} path={path}")
    if len(missing) > 20:
        lines.append(f"  ... {len(missing) - 20} more")
    return "\n".join(lines)


def wait_for_completion(
    config,
    wait_timeout=0,
    stall_timeout=600,
    poll_interval=10,
):
    cfg = normalize_config(config)
    chunks = build_chunk_list(cfg)
    total = len(chunks)
    if total == 0:
        print("done 0/0")
        return True

    wait_timeout = float(wait_timeout or 0)
    stall_timeout = float(stall_timeout or 0)
    poll_interval = max(0.01, float(poll_interval))
    started = time.monotonic()
    last_progress = started
    last_done = -1
    last_print = 0.0

    while True:
        missing = _missing_chunks(cfg, chunks)
        done = total - len(missing)
        now = time.monotonic()
        if done != last_done:
            last_progress = now
            last_done = done
            print(f"done {done}/{total}")
            last_print = now
        elif now - last_print >= poll_interval:
            print(f"done {done}/{total}")
            last_print = now

        if done == total:
            print("all partial LUTs present")
            return True

        if wait_timeout > 0 and now - started >= wait_timeout:
            raise WaitTimeoutError(
                _format_missing_message("wait timeout reached", missing, total),
                missing,
            )

        if stall_timeout > 0 and now - last_progress >= stall_timeout:
            raise WaitTimeoutError(
                _format_missing_message("stall timeout reached", missing, total),
                missing,
            )

        sleep_for = poll_interval
        if wait_timeout > 0:
            sleep_for = min(sleep_for, max(0.01, wait_timeout - (now - started)))
        if stall_timeout > 0:
            sleep_for = min(sleep_for, max(0.01, stall_timeout - (now - last_progress)))
        time.sleep(sleep_for)


def reduce_and_score(config, do_reduce=True, do_score=False):
    cfg = normalize_config(config)
    lut = None
    score = None
    if do_reduce:
        lut = combine_segment_lut_tile_zyx(
            cfg["z_range"],
            cfg["y_range"],
            cfg["x_range"],
            lut_output_path_format(cfg),
        )
        write_h5(combined_lut_path(cfg), lut)
        print(f"wrote combined LUT: {combined_lut_path(cfg)} ({len(lut)} nodes)")

    if do_score:
        if lut is None:
            lut = read_vol(combined_lut_path(cfg))
        graph = ERLGraph.from_npz(graph_path(cfg))
        score = score_graph_with_lut(
            graph,
            lut,
            merge_threshold=cfg["merge_threshold"],
            output_path=cfg["output_path"],
            lut_source=combined_lut_path(cfg),
        )
    return lut, score


def build_parser():
    parser = argparse.ArgumentParser(
        description="Map, reduce, and score ERL node LUTs from chunked segmentation files."
    )
    parser.add_argument("--config", required=True, help="YAML file with volume_eval_chunk block")
    parser.add_argument("--init-only", action="store_true", help="build graph and shared vertices, then exit")
    parser.add_argument("--chunk-index", type=int, help="flat chunk index to map")
    parser.add_argument("--chunk-range", default="", help="inclusive chunk range, e.g. 0-15 or 0-3,8")
    parser.add_argument("--parallel", type=int, help="map chunks locally with this many processes")
    parser.add_argument("--local", action="store_true", help="force local multiprocess backend")
    parser.add_argument("--sbatch", action="store_true", help="submit a SLURM array job")
    parser.add_argument("--wait", action="store_true", help="wait until every partial LUT file exists")
    parser.add_argument("--wait-timeout", type=float, default=0, help="max seconds to wait; 0 means no cap")
    parser.add_argument("--stall-timeout", type=float, default=600, help="seconds without progress before failing")
    parser.add_argument("--poll-interval", type=float, default=10, help="seconds between wait progress polls")
    parser.add_argument("--reduce", action="store_true", help="combine all partial LUTs")
    parser.add_argument("--score", action="store_true", help="score ERL from gt_graph.npz and seg_lut_all.h5")
    parser.add_argument("overrides", nargs="*", help="config overrides as key=value")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        cfg = build_config(args.config, args.overrides)
        backend = (
            "slurm"
            if args.sbatch
            else "multiprocess"
            if args.local
            else cfg.get("backend", "multiprocess")
        )
        cfg["backend"] = backend

        if args.init_only:
            init_workflow(cfg)
            return 0

        if args.sbatch:
            submit_sbatch(cfg, config_path=args.config)
            return 0

        explicit_chunk_indices = []
        if args.chunk_index is not None:
            explicit_chunk_indices.append(args.chunk_index)
        explicit_chunk_indices.extend(_chunk_indices_from_range(args.chunk_range))

        has_high_level_action = any(
            [args.wait, args.reduce, args.score, args.parallel is not None, args.local]
        )
        if (
            not explicit_chunk_indices
            and not has_high_level_action
            and os.environ.get("SLURM_ARRAY_TASK_ID") is not None
        ):
            explicit_chunk_indices.append(int(os.environ["SLURM_ARRAY_TASK_ID"]))

        if explicit_chunk_indices:
            run_chunk_indices(cfg, explicit_chunk_indices)
            if not any([args.wait, args.reduce, args.score]):
                return 0

        if args.parallel is not None or args.local:
            workers = args.parallel if args.parallel is not None else cfg["num_workers"]
            run_chunks_local(cfg, num_workers=workers)
            if not any([args.wait, args.reduce, args.score]):
                return 0

        if args.wait:
            wait_for_completion(
                cfg,
                wait_timeout=args.wait_timeout,
                stall_timeout=args.stall_timeout,
                poll_interval=args.poll_interval,
            )

        if args.reduce or args.score:
            reduce_and_score(cfg, do_reduce=args.reduce, do_score=args.score)
            return 0

        if backend == "slurm":
            submit_sbatch(cfg, config_path=args.config)
        elif backend == "multiprocess":
            run_chunks_local(cfg, num_workers=cfg["num_workers"])
        else:
            raise ValueError(f"Unknown backend {backend!r}")
        return 0
    except WaitTimeoutError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
