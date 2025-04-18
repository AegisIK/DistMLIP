import numpy as np
from matgl.distributed.subgraph_creation_fast import get_subgraphs_fast
import torch


class Distributed:
    def __init__(
        self,
        src_nodes,
        dst_nodes,
        markers,
        local_coords,
        global_ids,
        py_index_1,
        py_index_2,
        py_offsets,
        py_distances,
        line_src_nodes,
        line_dst_nodes,
        within_r_indices,
        line_markers,
        num_UDEs_per_partition,
        bond_mapping_DE_list,
        bond_mapping_UDE_list,
        L2G_DE_mapping_list,
        G2L_DE_mapping_list,
        local_center_atom_indices_list,
        use_bond_graph,
        total_num_nodes,
    ) -> None:
        self.src_nodes = src_nodes
        self.dst_nodes = dst_nodes
        self.markers = markers
        self.local_coords = local_coords
        self.global_ids = global_ids  # list of lists: (# GPUs, n), consists of the global atom graph node IDs
        self.py_index_1 = py_index_1
        self.py_index_2 = py_index_2
        self.py_offsets = py_offsets
        self.py_distances = py_distances
        self.line_src_nodes = line_src_nodes
        self.line_dst_nodes = line_dst_nodes
        self.within_r_indices = within_r_indices
        self.line_markers = line_markers
        self.num_UDEs_per_partition = num_UDEs_per_partition
        self.bond_mapping_DE_list = bond_mapping_DE_list
        self.bond_mapping_UDE_list = bond_mapping_UDE_list
        self.L2G_DE_mapping_list = (
            L2G_DE_mapping_list  # index i is the global id associated with local id i
        )
        self.G2L_DE_mapping_list = G2L_DE_mapping_list  # index j is the local id associated with global id j. <--- THIS IS NOT RELIABLE.
        self.local_center_atom_indices_list = local_center_atom_indices_list
        self.use_bond_graph = use_bond_graph

        self.num_partitions = len(src_nodes)
        self.total_num_edges = len(py_index_1)
        self.total_num_nodes = total_num_nodes

    @classmethod
    def create_distributed(
        cls,
        cart_coords,
        frac_coords,
        lattice_matrix,
        num_partitions,
        pbc,
        cutoff,
        three_body_cutoff,
        tol,
        use_bond_graph=False,
        num_threads=1,
    ):
        cart_coords = np.ascontiguousarray(cart_coords, dtype=float)
        frac_coords = np.ascontiguousarray(frac_coords, dtype=float)
        lattice_matrix = np.ascontiguousarray(lattice_matrix, dtype=float)

        (
            src_nodes,
            dst_nodes,
            markers,
            local_coords,
            global_ids,  # list of lists: (# GPUs, n), consists of the global atom graph node IDs
            py_index_1,
            py_index_2,
            py_offsets,
            py_distances,
            line_src_nodes,
            line_dst_nodes,
            within_r_indices,
            line_markers,
            num_UDEs_per_partition,
            bond_mapping_DE_list,
            bond_mapping_UDE_list,
            L2G_DE_mapping_list,  # index i is the global id associated with local id i
            G2L_DE_mapping_list,  # index j is the local id associated with global id j. <--- THIS IS NOT RELIABLE.
            local_center_atom_indices_list,
        ) = get_subgraphs_fast(
            cart_coords,
            float(cutoff),
            pbc,
            lattice_matrix,
            num_partitions,
            float(three_body_cutoff),
            tol,
            num_threads,
            use_bond_graph,
            frac_coords,
        )

        # Set up markers arrays
        markers_tmp = []
        for i, marker in enumerate(markers):
            markers_tmp.append(np.append(marker, len(local_coords[i])))
        markers = markers_tmp

        if use_bond_graph:
            assert (
                len(line_markers) == 0 or len(line_markers) == num_partitions
            ), f"Length of line_markers is {len(line_markers)} which is not {num_partitions} or 0."
            line_markers_tmp = []
            for i in range(len(line_markers)):
                line_markers_tmp.append(
                    np.append(line_markers[i], num_UDEs_per_partition[i])
                )
            line_markers = line_markers_tmp
        else:
            line_markers = None

        return cls(
            src_nodes,
            dst_nodes,
            markers,
            local_coords,
            global_ids,
            py_index_1,
            py_index_2,
            py_offsets,
            py_distances,
            line_src_nodes,
            line_dst_nodes,
            within_r_indices,
            line_markers,
            num_UDEs_per_partition,
            bond_mapping_DE_list,
            bond_mapping_UDE_list,
            L2G_DE_mapping_list,
            G2L_DE_mapping_list,
            local_center_atom_indices_list,
            use_bond_graph,
            len(cart_coords),
        )

    def aggregate(self, features_to_aggregate, device="cpu", aggregate_dim=None):
        """
        Aggregates the features of features_to_aggregate into a single tensor on gpu_to_aggregate_to
        """

        if not aggregate_dim:
            aggregate_dim = self.total_num_nodes

        combined_feats = torch.empty(
            (aggregate_dim,) + tuple(features_to_aggregate[0].shape[1:]),
            device=device,
            dtype=features_to_aggregate[0].dtype,
        )

        to_from_cutoff = self.num_partitions + 1

        for partition_i in range(self.num_partitions):
            this_global_ids = self.global_ids[partition_i]
            this_cutoff = self.markers[partition_i][to_from_cutoff]

            this_partition_global_ids = this_global_ids[:this_cutoff]

            combined_feats[this_partition_global_ids] = features_to_aggregate[
                partition_i
            ][:this_cutoff].to(device)

        return combined_feats

    def transfer_nodes(self, features, markers):
        """
        Transfers node features between all partitions such that each partition has the most up to date border node features
        """
        assert len(features) == self.num_partitions

        for curr in range(self.num_partitions):
            for to in range(self.num_partitions):
                if curr == to:
                    continue
                from_start = markers[curr][1 + to]
                from_end = markers[curr][1 + to + 1]

                to_start = markers[to][1 + self.num_partitions + curr]
                to_end = markers[to][1 + self.num_partitions + curr + 1]

                if (
                    from_start != from_end
                ):  # Only attempt to transfer if there is stuff to transfer
                    features[to][to_start:to_end] = features[curr][from_start:from_end]

        return features

    def atom_transfer(self, features):
        """
        Updates the atom graph node features such that each gpu has the most up to date features.
        """

        return self.transfer_nodes(features, self.markers)

    def bond_transfer(self, features):
        """
        Updates the bond graph node features such that each gpu has the most up to date features.
        """

        return self.transfer_nodes(features, self.line_markers)

    def aggregate_atom_edge(self, atom_edge_features, gpu_to_aggregate_to="cpu"):
        """
        Aggregates the edge features within atom graph into a single tensor
        """

        aggregated_feats = torch.empty(
            (self.total_num_edges,) + tuple(atom_edge_features[0].shape[1:]),
            device=gpu_to_aggregate_to,
            dtype=atom_edge_features[0].dtype,
        )

        for i, m in enumerate(self.L2G_DE_mapping_list):
            aggregated_feats[m] = atom_edge_features[i].to(gpu_to_aggregate_to)

        return aggregated_feats

    def aggregate_bond_node(self, bond_node_features, gpu_to_aggregate_to):
        """
        Aggregates the node features within bond graph into a single tensor with dimension (number of atom edges, *feature_dim)

        Edge indices within atom graph that don't correspond to nodes within bond graph will have their features defaulted to 0
        """

        placeholder_dims = [
            tuple([len(self.src_nodes[i])] + bond_node_features[0].shape[1:])
            for i in range(self.num_partitions)
        ]
        atom_edge_features_placeholders = [
            torch.zeros(
                placeholder_dims,
                device=gpu_to_aggregate_to,
                dtype=bond_node_features[0].dtype,
            )
        ]

        for i in range(self.num_partitions):
            atom_edge_features_placeholders[i][self.bond_mapping_DE_list[i]] = (
                bond_node_features[i][self.bond_mapping_UDE_list[i]]
            )

        return self.aggregate_atom_edge(
            atom_edge_features_placeholders, gpu_to_aggregate_to=gpu_to_aggregate_to
        )

    def num_atoms(self, partition):
        return len(self.local_coords[partition])

    def num_bonds(self, partition):
        assert self.use_bond_graph, "num_bonds only works when bond graph is enabled"
        return self.line_markers[partition][-1]

    def global_to_local_nodes(
        self, global_node_features, partition, device="cpu", inplace=False
    ):
        if inplace:
            return global_node_features[self.global_ids[partition]].to(device)
        return global_node_features.clone()[self.global_ids[partition]].to(device)

    def global_to_local_edges(
        self, global_edge_features, partition, device="cpu", inplace=False
    ):
        if inplace:
            return global_edge_features[self.L2G_DE_mapping_list[partition]].to(device)
        return global_edge_features.clone()[self.L2G_DE_mapping_list[partition]].to(
            device
        )

    def edge_to_bond(
        self,
        edge_features,
        partition,
        device="cpu",
        inplace=False,
        bond_node_features=None,
    ):
        """
        Transfers edge features in atom graph to node features in bond graph

        edge_features - edge features for the current partition
        inplace - boolean - determines whether or not we have a bond_node_features tensor to place the edge features into
        TODO: add some basic checks to make sure that the edge features are for the current partition
        """
        if inplace:
            assert (
                bond_node_features
            ), "bond_node_features cannot be None if inplace is True"
            bond_node_features[partition][self.bond_mapping_UDE_list[partition]] = (
                edge_features[partition][self.bond_mapping_DE_list[partition]]
            )
            return

        bond_features = torch.zeros(
            (self.num_bonds(partition),) + tuple(edge_features.shape[1:]), device=device
        )
        bond_features[self.bond_mapping_UDE_list[partition]] = edge_features[
            self.bond_mapping_DE_list[partition]
        ]

        return bond_features

    def bond_to_edge(self, bond_node_features, atom_edge_features, partition):
        """
        Transfers node features in bond graph to edge features in atom graph
        """
        atom_edge_features[partition][self.bond_mapping_DE_list[partition]] = (
            bond_node_features[partition][self.bond_mapping_UDE_list[partition]]
        )
