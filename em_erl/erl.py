import inspect
from dataclasses import dataclass

import numpy as np


_DATACLASS_KWARGS = {"slots": True} if "slots" in inspect.signature(dataclass).parameters else {}


class SkeletonScore:
    def __init__(self):
        self.omitted = 0
        self.split = 0
        self.merged_seg = []
        self.merged_seg_num = []
        self.merged_mask = []
        self.merged_mask_num = []
        self.correct_seg = []
        self.correct_len = []

    def print(self):
        if self.omitted > 0:
            print(f"# omitted edge: {self.omitted}")
        if self.split > 0:
            print(f"# split edge: {self.split}")
        if len(self.merged_seg) > 0:
            print(f"merged seg id: {self.merged_seg}")
            print(f"merged voxel count: {self.merged_seg_num}")
        if len(self.merged_mask) > 0:
            print(f"merged mask seg id: {self.merged_mask}")
            print(f"merged voxel count: {self.merged_mask_num}")
        if len(self.correct_seg) > 0:
            print(f"correct seg id: {self.correct_seg}")
            print(f"count skel length: {self.correct_len}")


class ERLScore:
    def __init__(self, skeleton_id=None, skeleton_len=None, verbose=True):
        self.skeleton_id = skeleton_id
        self.skeleton_len = skeleton_len
        self.skeleton_erl = None
        self.erl = None
        self.erl_intervals = None
        if verbose:
            self.skeleton_score = None
            self.merged_seg = {}
            self.merged_mask = []

    def compute_erl(self, erl_intervals=None):
        """
        Compute the Error Rate Length (ERL) scores based on ERLScore and intervals.

        Args:
            erl_intervals (list, optional): Intervals for ERL computation. Defaults to None.

        Returns:
            None
        """
        self.erl_intervals = erl_intervals
        num_skel = len(self.skeleton_len)
        len_total = sum(self.skeleton_len)
        erl_gt = self.skeleton_len * self.skeleton_len
        erl_pred = self.skeleton_len * self.skeleton_erl
        if erl_intervals is not None:
            self.erl = np.zeros([len(erl_intervals), 3])
            self.erl[0] = [
                erl_pred.sum() / len_total,
                erl_gt.sum() / len_total,
                num_skel,
            ]
            for i in range(1, len(erl_intervals)):
                skeleton_index = (self.skeleton_len >= erl_intervals[i - 1]) * (
                    self.skeleton_len < erl_intervals[i]
                )
                len_interval = self.skeleton_len[skeleton_index].sum()
                self.erl[i, 0] = erl_pred[skeleton_index].sum() / len_interval
                self.erl[i, 1] = erl_gt[skeleton_index].sum() / len_interval
                self.erl[i, 2] = skeleton_index.sum()
        else:
            self.erl = np.array(
                [
                    erl_pred.sum() / len_total,
                    erl_gt.sum() / len_total,
                    num_skel,
                ]
            )

    def print_erl(self, erl=None, erl_intervals=None):
        erl = self.erl if erl is None else erl
        erl_intervals = self.erl_intervals if erl_intervals is None else erl_intervals

        if erl.ndim == 1:
            erl = erl.reshape(1, -1)
        for i in range(erl.shape[0]):
            if i == 0:
                print("all skel")
            elif erl_intervals is not None:
                print(f"gt skel range: {erl_intervals[i-1]} - {erl_intervals[i]}")
            print(f"ERL\t: {erl[i, 0]:.2f}")
            print(f"gt ERL\t: {erl[i, 1]:.2f}")
            print(f"#skel\t: {int(erl[i, 2]):d}")
            print("-----------------")


@dataclass(**_DATACLASS_KWARGS)
class ERLGraph:
    """Compact skeleton graph representation with flat edge arrays."""

    skeleton_id: np.ndarray | None = None
    skeleton_len: np.ndarray | None = None
    node_skeleton_index: np.ndarray | None = None
    node_coords_zyx: np.ndarray | None = None
    edge_u: np.ndarray | None = None
    edge_v: np.ndarray | None = None
    edge_len: np.ndarray | None = None
    edge_ptr: np.ndarray | None = None

    SCHEMA_VERSION = 2
    _SCHEMA_KEY = "erl_graph_schema_version"

    def __post_init__(self):
        fields = [
            self.skeleton_id,
            self.skeleton_len,
            self.node_skeleton_index,
            self.node_coords_zyx,
            self.edge_u,
            self.edge_v,
            self.edge_len,
            self.edge_ptr,
        ]
        present = [x is not None for x in fields]
        if any(present) and not all(present):
            raise ValueError("Provide all ERLGraph arrays or none of them.")
        if all(present):
            self.validate()

    @property
    def num_skeletons(self):
        return 0 if self.skeleton_id is None else int(len(self.skeleton_id))

    @property
    def num_nodes(self):
        return 0 if self.node_skeleton_index is None else int(len(self.node_skeleton_index))

    @property
    def num_edges(self):
        return 0 if self.edge_u is None else int(len(self.edge_u))

    def validate(self):
        self.skeleton_id = np.asarray(self.skeleton_id)
        self.skeleton_len = np.asarray(self.skeleton_len)
        self.node_skeleton_index = np.asarray(self.node_skeleton_index)
        self.node_coords_zyx = np.asarray(self.node_coords_zyx)
        self.edge_u = np.asarray(self.edge_u)
        self.edge_v = np.asarray(self.edge_v)
        self.edge_len = np.asarray(self.edge_len)
        self.edge_ptr = np.asarray(self.edge_ptr)

        if self.skeleton_id.ndim != 1:
            raise ValueError("skeleton_id must be 1D")
        if self.skeleton_len.ndim != 1:
            raise ValueError("skeleton_len must be 1D")
        if len(self.skeleton_id) != len(self.skeleton_len):
            raise ValueError("skeleton_id and skeleton_len must have the same length")
        if self.node_skeleton_index.ndim != 1:
            raise ValueError("node_skeleton_index must be 1D")
        if self.node_coords_zyx.ndim != 2 or self.node_coords_zyx.shape[1] != 3:
            raise ValueError("node_coords_zyx must have shape [N, 3]")
        if len(self.node_skeleton_index) != len(self.node_coords_zyx):
            raise ValueError("node arrays must have the same length")
        if self.edge_u.ndim != 1 or self.edge_v.ndim != 1 or self.edge_len.ndim != 1:
            raise ValueError("edge arrays must be 1D")
        if not (len(self.edge_u) == len(self.edge_v) == len(self.edge_len)):
            raise ValueError("edge_u, edge_v and edge_len must have the same length")
        if self.edge_ptr.ndim != 1:
            raise ValueError("edge_ptr must be 1D")
        if len(self.edge_ptr) != len(self.skeleton_id) + 1:
            raise ValueError("edge_ptr must have length num_skeletons + 1")
        if len(self.edge_ptr) == 0 or int(self.edge_ptr[0]) != 0:
            raise ValueError("edge_ptr must start at 0")
        if int(self.edge_ptr[-1]) != len(self.edge_u):
            raise ValueError("edge_ptr[-1] must equal number of edges")
        if np.any(np.diff(self.edge_ptr) < 0):
            raise ValueError("edge_ptr must be non-decreasing")
        if len(self.edge_u) > 0:
            node_num = len(self.node_skeleton_index)
            if self.edge_u.min() < 0 or self.edge_v.min() < 0:
                raise ValueError("edge indices must be non-negative")
            if self.edge_u.max() >= node_num or self.edge_v.max() >= node_num:
                raise ValueError("edge indices out of bounds")
        if len(self.node_skeleton_index) > 0:
            if self.node_skeleton_index.min() < 0:
                raise ValueError("node_skeleton_index must be non-negative")
            if self.node_skeleton_index.max() >= len(self.skeleton_id):
                raise ValueError("node_skeleton_index out of bounds")

        return self

    def get_nodes_position(self, resolution=None):
        # return voxel position of all nodes
        assert self.node_coords_zyx is not None
        if resolution is None:
            position = self.node_coords_zyx
        else:
            position = self.node_coords_zyx / np.asarray(resolution)
        return position.astype(int)

    @classmethod
    def from_npz(cls, input_file):
        data = np.load(input_file, allow_pickle=False)
        if cls._SCHEMA_KEY in data:
            version = int(np.asarray(data[cls._SCHEMA_KEY]).item())
            if version != cls.SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported ERLGraph schema version {version}. "
                    f"Expected {cls.SCHEMA_VERSION}."
                )
            return cls(
                skeleton_id=data["skeleton_id"],
                skeleton_len=data["skeleton_len"],
                node_skeleton_index=data["node_skeleton_index"],
                node_coords_zyx=data["node_coords_zyx"],
                edge_u=data["edge_u"],
                edge_v=data["edge_v"],
                edge_len=data["edge_len"],
                edge_ptr=data["edge_ptr"],
            )
        return cls._from_legacy_npz(data)

    @classmethod
    def _from_legacy_npz(cls, data):
        skeleton_id = data["skeleton_id"]
        skeleton_len = data["skeleton_len"]
        nodes = data["nodes"]
        edge_keys = sorted(
            (k for k in data.keys() if k.startswith("edges_")),
            key=lambda k: int(k.split("_")[1]),
        )
        edges = [data[k] for k in edge_keys]
        return cls.from_legacy_arrays(skeleton_id, skeleton_len, nodes, edges)

    @classmethod
    def from_legacy_arrays(cls, skeleton_id, skeleton_len, nodes, edges):
        nodes = np.asarray(nodes)
        if nodes.ndim != 2 or nodes.shape[1] != 4:
            raise ValueError("Legacy nodes must have shape [N, 4]")

        edge_counts = np.array([0 if e is None else len(e) for e in edges], dtype=np.int64)
        edge_ptr = np.zeros(len(edge_counts) + 1, dtype=np.int64)
        edge_ptr[1:] = np.cumsum(edge_counts)

        if edge_ptr[-1] > 0:
            edge_arr = np.vstack([np.asarray(e) for e in edges if e is not None and len(e) > 0])
            edge_u = edge_arr[:, 0].astype(np.uint32, copy=False)
            edge_v = edge_arr[:, 1].astype(np.uint32, copy=False)
            edge_len = edge_arr[:, 2].astype(np.float32, copy=False)
        else:
            edge_u = np.zeros(0, dtype=np.uint32)
            edge_v = np.zeros(0, dtype=np.uint32)
            edge_len = np.zeros(0, dtype=np.float32)

        return cls(
            skeleton_id=np.asarray(skeleton_id),
            skeleton_len=np.asarray(skeleton_len),
            node_skeleton_index=nodes[:, 0].astype(np.uint32, copy=False),
            node_coords_zyx=nodes[:, 1:4].copy(),
            edge_u=edge_u,
            edge_v=edge_v,
            edge_len=edge_len,
            edge_ptr=edge_ptr.astype(np.uint64, copy=False),
        )

    def save_npz(self, output_file):
        self.validate()
        np.savez_compressed(
            output_file,
            **{
                self._SCHEMA_KEY: np.array(self.SCHEMA_VERSION, dtype=np.uint16),
                "skeleton_id": self.skeleton_id,
                "skeleton_len": self.skeleton_len,
                "node_skeleton_index": self.node_skeleton_index,
                "node_coords_zyx": self.node_coords_zyx,
                "edge_u": self.edge_u,
                "edge_v": self.edge_v,
                "edge_len": self.edge_len,
                "edge_ptr": self.edge_ptr,
            },
        )

    def print_info(self):
        print(f"Number of skeletons: {len(self.skeleton_id)}")
        if len(self.skeleton_len) == 0:
            print("Skeleton length (min, max, mean): (0.00, 0.00, 0.00)")
            return
        print(
            "Skeleton length (min, max, mean): "
            f"({self.skeleton_len.min():.2f}, {self.skeleton_len.max():.2f}, {self.skeleton_len.mean():.2f})"
        )

    def edge_span(self, skeleton_index):
        start = int(self.edge_ptr[skeleton_index])
        end = int(self.edge_ptr[skeleton_index + 1])
        return start, end

    def edge_skeleton_index(self):
        if self.num_skeletons == 0:
            return np.zeros(0, dtype=np.uint32)
        edge_counts = np.diff(self.edge_ptr).astype(np.int64, copy=False)
        skeleton_index = np.arange(self.num_skeletons, dtype=np.uint32)
        return np.repeat(skeleton_index, edge_counts)


def skel_to_erlgraph(
    skeletons,
    skeleton_resolution=None,
    length_threshold=0,
    sample_ratio=1,
    coord_dtype=np.float32,
    index_dtype=np.uint32,
    edge_len_dtype=np.float32,
):
    # if skeleton_resolution is None: skeleton nodes are already in physical units
    # else: skeleton nodes are in voxel units and will be scaled to physical units
    skeleton_ids = np.array(list(skeletons.keys()))
    if sample_ratio < 1:
        rand_idx = np.random.permutation(len(skeleton_ids))
        rand_idx = rand_idx[: int(len(skeleton_ids) * sample_ratio)]
        skeleton_ids = skeleton_ids[rand_idx]

    scale = None
    if skeleton_resolution is not None:
        scale = np.asarray(skeleton_resolution, dtype=np.float64)

    kept_ids = []
    kept_len = []
    node_skel_chunks = []
    node_coord_chunks = []
    edge_u_chunks = []
    edge_v_chunks = []
    edge_len_chunks = []
    edge_ptr = [0]

    node_offset = 0
    edge_offset = 0

    for skeleton_id in skeleton_ids:
        skeleton = skeletons[skeleton_id]
        edges = np.asarray(skeleton.edges)
        if len(edges) == 0:
            continue
        edges = edges.astype(np.int64, copy=False)

        verts = np.asarray(skeleton.vertices)
        verts_phys = verts.astype(np.float64, copy=False)
        if scale is not None:
            verts_phys = verts_phys * scale

        diff = verts_phys[edges[:, 0]] - verts_phys[edges[:, 1]]
        edges_len = np.linalg.norm(diff, axis=1)
        total_len = float(edges_len.sum())
        if total_len < length_threshold:
            continue

        skel_index = len(kept_ids)
        num_nodes = verts.shape[0]
        num_edges = edges.shape[0]

        kept_ids.append(skeleton_id)
        kept_len.append(total_len)

        node_skel_chunks.append(np.full(num_nodes, skel_index, dtype=index_dtype))
        node_coord_chunks.append(verts_phys.astype(coord_dtype, copy=False))

        edge_u_chunks.append((edges[:, 0] + node_offset).astype(index_dtype, copy=False))
        edge_v_chunks.append((edges[:, 1] + node_offset).astype(index_dtype, copy=False))
        edge_len_chunks.append(edges_len.astype(edge_len_dtype, copy=False))

        node_offset += num_nodes
        edge_offset += num_edges
        edge_ptr.append(edge_offset)

    if kept_ids:
        node_skeleton_index = np.concatenate(node_skel_chunks)
        node_coords_zyx = np.vstack(node_coord_chunks)
        edge_u = np.concatenate(edge_u_chunks)
        edge_v = np.concatenate(edge_v_chunks)
        edge_len = np.concatenate(edge_len_chunks)
    else:
        node_skeleton_index = np.zeros(0, dtype=index_dtype)
        node_coords_zyx = np.zeros((0, 3), dtype=coord_dtype)
        edge_u = np.zeros(0, dtype=index_dtype)
        edge_v = np.zeros(0, dtype=index_dtype)
        edge_len = np.zeros(0, dtype=edge_len_dtype)

    return ERLGraph(
        skeleton_id=np.asarray(kept_ids),
        skeleton_len=np.asarray(kept_len, dtype=np.float64),
        node_skeleton_index=node_skeleton_index,
        node_coords_zyx=node_coords_zyx,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_len=edge_len,
        edge_ptr=np.asarray(edge_ptr, dtype=np.uint64),
    )
