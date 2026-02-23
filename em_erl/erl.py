import numpy as np

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


class ERLGraph:
    # The ERLGraph class is a lightweight class of ground truth skeletons for efficient ERL evaluation.
    def __init__(
        self,
        input_file=None,
    ):
        if input_file is None:
            self.skeleton_id = None
            self.skeleton_len = None
            # all nodes in one matrix: [N, 4] (skel_id, z, y, x)
            self.nodes = None
            # all edges in one list: [[M, 3]] (node_1, node_2, length)
            self.edges = None
        else:
            self.load_npz(input_file)

    def get_nodes_position(self, resolution=None):
        # return voxel position of all nodes
        assert self.nodes is not None
        position = (
            self.nodes[:, 1:] if resolution is None else self.nodes[:, 1:] // resolution
        )
        return position.astype(int)

    def load_npz(self, input_file):
        """
        The function `load_npz` loads node and edge data from npz files and initializes viewers.

        :param node_npz_file: The parameter `node_npz_file` is the file path to the .npz file that
        contains the data for the nodes
        :param edge_npz_file: The `edge_npz_file` parameter is a file path to a NumPy compressed sparse
        matrix file (.npz) that contains the edge data
        """
        data = np.load(input_file)
        self.edges = [None] * (len(data.keys()) - 3)
        for i, key in enumerate(data):
            if i == 0:
                self.skeleton_id = data[key]
            elif i == 1:
                self.skeleton_len = data[key]
            elif i == 2:
                self.nodes = data[key]
            else:
                self.edges[i - 3] = data[key]

    def save_npz(self, output_file):
        assert self.nodes is not None
        assert self.edges is not None
        output = [self.skeleton_id] + [self.skeleton_len] + [self.nodes] + self.edges
        np.savez_compressed(output_file, *output)

    def print_info(self):
        print(f"Number of skeletons: {len(self.skeleton_id)}")
        print(
            f"Skeleton length (min, max, mean): ({self.skeleton_len.min():.2f}, {self.skeleton_len.max():.2f}, {self.skeleton_len.mean():.2f})"
        )


def skel_to_erlgraph(
    skeletons,
    skeleton_resolution=None,
    length_threshold=0,
    sample_ratio=1,
    node_dtype=np.uint32,
    edge_dtype=np.float32,
):
    # if skeleton_resolution is None: skeleton nodes are in physical units
    # else: skeleton nodes are in voxel units
    graph = ERLGraph()
    graph.skeleton_id = np.array(list(skeletons.keys()))
    if sample_ratio < 1:
        # need to subsample the data
        rand_idx = np.random.permutation(len(graph.skeleton_id))
        rand_idx = rand_idx[: int(len(graph.skeleton_id) * sample_ratio)]
        graph.skeleton_id = graph.skeleton_id[rand_idx]
    skeleton_num = len(graph.skeleton_id)
    graph.skeleton_len = np.zeros(skeleton_num)

    count = 0
    graph.nodes = [np.zeros([0, 4], node_dtype) for _ in range(skeleton_num)]
    graph.edges = [None] * skeleton_num

    count_skel = 0

    for i, skeleton_id in enumerate(graph.skeleton_id):
        skeleton = skeletons[skeleton_id]
        if len(skeleton.edges) == 0:
            continue
        node_arr = skeleton.vertices.astype(node_dtype)
        if skeleton_resolution is not None:
            node_arr = node_arr * np.array(skeleton_resolution).astype(node_dtype)
        edges_len = np.linalg.norm(
            node_arr[skeleton.edges[:, 0]] - node_arr[skeleton.edges[:, 1]].astype(int),
            axis=1,
        )
        graph.skeleton_len[i] = edges_len.sum()
        if graph.skeleton_len[i] >= length_threshold:
            num_arr = node_arr.shape[0]
            ind_arr = count_skel * np.ones([num_arr, 1], node_dtype)
            graph.nodes[i] = np.hstack([ind_arr, node_arr])

            graph.edges[i] = np.hstack(
                [skeleton.edges + count, edges_len.reshape(-1, 1)]
            )
            count += num_arr
            count_skel += 1

    graph.nodes = np.vstack(graph.nodes)
    if length_threshold > 0 and graph.skeleton_len.min() < length_threshold:
        # need to remove small seg_ids
        graph.skeleton_id = graph.skeleton_id[graph.skeleton_len >= length_threshold]
        graph.edges = [x for x in graph.edges if x is not None]
        graph.skeleton_len = graph.skeleton_len[graph.skeleton_len >= length_threshold]

    return graph


def convert_networkx(self, nx_graph):
    """
    The function loads a graph into the object, ensuring that the graph nodes have the same
    attributes and storing the node and edge data in appropriate data structures.

    :param graph: The `graph` parameter is an object that represents a graph. It contains
    information about the nodes and edges of the graph
    """
    node_attributes = (["skeleton_id", "z", "y", "x"],)
    edge_attributes = (["length", "e1", "e2"],)
    assert len(nx_graph.nodes) > 0
    # assert every node has the same attributes
    assert list(nx_graph.nodes) == list(range(len(nx_graph.nodes)))

    nodes = {key: [] for key in self.node_attributes}

    minval = np.inf
    maxval = 0
    for node in nx_graph.nodes:
        node = nx_graph.nodes[node]
        for key in self.node_attributes:
            assert key in node
            nodes[key].append(node[key])
            maxval = max(maxval, node[key])
            minval = min(minval, node[key])
    assert minval >= np.iinfo(self.node_dtype).min
    assert maxval <= np.iinfo(self.node_dtype).max
    assert len({len(nodes[key]) for key in nodes}) == 1

    self._nodes = np.stack(
        [np.array(nodes[key]) for key in self.node_attributes], axis=1
    ).astype(self.node_dtype)

    edges = sp.dok_matrix(
        (len(nx_graph.nodes), len(nx_graph.nodes)), dtype=self.edge_dtype
    )
    for edge_0, edge_1, data in nx_graph.edges(data=True):
        edge = tuple(sorted([edge_0, edge_1]))
        edges[edge] = data[self.edge_attribute] if self.edge_attribute in data else -1

    self._edges = edges
    self.init_viewers()
