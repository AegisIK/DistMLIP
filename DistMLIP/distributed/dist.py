from __future__ import annotations

import numpy as np
from DistMLIP.distributed.subgraph_creation_fast import get_subgraphs_fast
import torch
from typing import List, Tuple, Optional, Union


class Distributed:
    "Distributed Graph for parallelized MLIP inference"

    def __init__(
        self,
        src_nodes: List[np.ndarray],
        dst_nodes: List[np.ndarray],
        markers: List[np.ndarray],
        local_coords: List[np.ndarray],
        global_ids: List[np.ndarray],
        py_index_1: np.ndarray,
        py_index_2: np.ndarray,
        py_offsets: np.ndarray,
        py_distances: np.ndarray,
        line_src_nodes: List[np.ndarray],
        line_dst_nodes: List[np.ndarray],
        within_r_indices: List[np.ndarray],
        line_markers: List[np.ndarray] | None,
        num_UDEs_per_partition: List[int],
        bond_mapping_DE_list: List[np.ndarray],
        bond_mapping_UDE_list: List[np.ndarray],
        L2G_DE_mapping_list: List[np.ndarray],
        G2L_DE_mapping_list: List[np.ndarray],
        local_center_atom_indices_list: List[np.ndarray],
        use_bond_graph: bool,
        total_num_nodes: int,
    ) -> None:
        """
        Initializes the Distributed Graph.

        Args:
            src_nodes (list): List of numpy arrays, where each array contains the
               source node indices (local) for edges in that partition's atom graph.
            dst_nodes (list): List of numpy arrays, where each array contains the
               destination node indices (local) for edges in that partition's atom
               graph.
            markers (list): List of numpy arrays. Each array marks boundaries for
               different types of nodes within a partition (e.g., pure nodes, border
               nodes shared with specific other partitions).
               The structure of 1 numpy array in the list, for an n partition system, is:
               [0, pure nodes, to_0, to_1, ..., to_n, from_0, from_1, ... from_n]. The
               elememnt in the list is the total number of atoms within a partition
               (including border nodes). Used in transferring node information
               between atom graphs.
            local_coords (list): List of numpy arrays, each holding the Cartesian
               coordinates of nodes (atoms) within atom graph within that partition
               (including border nodes).
            global_ids (list): List of numpy arrays. Each array maps local node indices
               in a partition to their original global node IDs. Therefore,
               global_ids[x][i] refers to the global id corresponding to local node
               i in partition x.
            py_index_1 (np.ndarray): Global source node indices for all edges across
               all partitions (concatenated) for atom graph. Often corresponds to the full,
               non-distributed graph's edge index.
            py_index_2 (np.ndarray): Global destination node indices for all edges
               across all partitions (concatenated) for atom graph. Often corresponds to the full,
               non-distributed graph's edge index.
            py_offsets (np.ndarray): Global offset vectors for periodic boundary
               conditions for all edges in atom graph.
            py_distances (np.ndarray): Global distances for all edges in atom graph.
            line_src_nodes (list): List of numpy arrays, source nodes for the edges in bond
               graph (if use_bond_graph is True).
            line_dst_nodes (list): List of numpy arrays, destination nodes for the edges in bond
               graph (if use_bond_graph is True).
            within_r_indices (list): Global edge indices (corresponding to py_index_1 and py_index_2)
               of all edges that are within the bond cutoff.
            line_markers (list): List of numpy arrays, similar to `markers` but for the
               bond graph nodes.
            num_UDEs_per_partition (list): Number of nodes in bond graph in each
               partition.
            bond_mapping_DE_list (list): List of numpy arrays. Maps directed edge
               indices (in atom graph) to their corresponding indices within the local
               partition's bond graph (bond graph nodes). To be used in
               conjunction with bond_mapping_UDE_list. See the `edge_to_bond` method
               for reference.
            bond_mapping_UDE_list (list): List of numpy arrays. Maps unique directed
               edge indices (bond graph nodes) back to their corresponding indices
               within the local partition's atom graph (atom graph edges). To be used in conjunction
               with bond_mapping_DE_list. See the `edge_to_bond` method for reference.
            L2G_DE_mapping_list (list): List of numpy arrays. Maps local directed edge
               indices (atom graph edges) in a partition to their global edge indices.
               `L2G_DE_mapping_list[partition_idx][local_edge_idx] = global_edge_idx`.
            G2L_DE_mapping_list (list): List of numpy arrays. Maps global directed edge
               indices to local directed edge indices. Note: This is deprecated and not reliable.
               Do not use this variable.
            local_center_atom_indices_list (list): List of numpy arrays, containing indices of
               the 'center' atoms (atom graph node) for each bond graph edge. This refers to the
               atom that is at the center of each angle represented by an edge within the bond graph.
            use_bond_graph (bool): Flag indicating if bond graph information is
               included and should be used.
            total_num_nodes (int): The total number of nodes (atoms) in the original,
               non-distributed atom graph.
        """
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

    @staticmethod
    def cartesian_to_wrapped_fractional(positions_cartesian, lattice, pbc):
        """
        Convert Cartesian coordinates to wrapped fractional coordinates.

        Parameters
        ----------
        positions_cartesian : np.ndarray
            Array of Cartesian positions, shape (N, 3).
        lattice : np.ndarray
            Lattice matrix, shape (3, 3). Rows are lattice vectors.

        Returns
        -------
        positions_fractional : np.ndarray
            Array of wrapped fractional positions, shape (N, 3).
        """
        # Convert to fractional
        positions_fractional = np.linalg.solve(lattice.T, np.transpose(positions_cartesian)).T
        
        for i, periodic in enumerate(pbc):
            if periodic:
                positions_fractional[:, i] %= 1.0
                positions_fractional[:, i] %= 1.0
        
        return positions_fractional

    @classmethod
    def create_distributed(
        cls,
        cart_coords: np.ndarray,
        frac_coords: np.ndarray,
        lattice_matrix: np.ndarray,
        num_partitions: int,
        pbc: np.ndarray,
        cutoff: float,
        three_body_cutoff: float=0,
        tol: float=1e-8,
        use_bond_graph: bool = False,
        num_threads: int = 1,
    ) -> "Distributed":
        """
        Class method to partition a graph and create a Distributed instance.

        This method takes the fundamental description of a periodic structure
        (coordinates, lattice, PBC flags) and partitioning parameters, then calls
        the `get_subgraphs_fast` function to perform the actual partitioning
        and generate the necessary data structures for distributed computation.

        Args:
            cart_coords (np.ndarray): Cartesian coordinates of all atoms (N x 3).
            frac_coords (np.ndarray): Fractional coordinates of all atoms (N x 3). Fractional coordinates should be wrapped.
            lattice_matrix (np.ndarray): Lattice vectors (3 x 3).
            num_partitions (int): The desired number of partitions (e.g., number of GPUs).
            pbc (np.ndarray): Periodic boundary conditions along each lattice vector direction.
            cutoff (float): The cutoff radius for finding neighbors (atom graph edges).
            three_body_cutoff (float): The cutoff radius for three-body interactions (if applicable).
            tol (float): Tolerance for distance calculations.
            use_bond_graph (bool, optional): Whether to generate data structures for the bond graph (line graph). Defaults to False.
            num_threads (int, optional): Number of threads to use for parallelization within `get_subgraphs_fast`. Defaults to 1.

        Returns:
            Distributed: A new instance of the Distributed class containing the partitioned graph data.
        """
        cart_coords = np.ascontiguousarray(cart_coords, dtype=float)
        frac_coords = np.ascontiguousarray(frac_coords, dtype=float)
        lattice_matrix = np.ascontiguousarray(lattice_matrix, dtype=float)

        (
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
            L2G_DE_mapping_list,  # index i is the global id associated with local id i
            G2L_DE_mapping_list,
            # index j is the local id associated with global id j. <--- THIS IS NOT RELIABLE.
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

    def aggregate(
        self,
        features_to_aggregate: List[torch.Tensor],
        device: Union[str, torch.device] = "cpu",
        aggregate_dim: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Aggregates non-border node features from each partition into a global tensor.

        Selects features for non-border nodes from each partition's local
        feature tensor and places them into a global tensor using global IDs.

        Args:
            features_to_aggregate: List of local node feature tensors, one
                per partition.
            device: Target device for the aggregated tensor. Defaults to "cpu".
            aggregate_dim: Size of the first dimension of the output tensor.
                Defaults to `self.total_num_nodes`.

        Returns:
            A single tensor with aggregated core node features, indexed globally.
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

    def transfer_nodes(
        self, features: List[torch.Tensor], markers: List[np.ndarray]
    ) -> List[torch.Tensor]:
        """
        Transfers border node features between partitions (in-place).

        Updates border node features in receiving partitions with computed
        features from the source partition's corresponding border nodes. Uses
        the `markers` array to identify data slices for transfer.

        Args:
            features: List of feature tensors (one per partition), modified
                in-place.
            markers: Marker arrays (`self.markers` or `self.line_markers`)
                defining border regions.

        Returns:
            The updated list of feature tensors (modified in-place).
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

                # Only attempt to transfer if there is stuff to transfer
                if from_start != from_end:
                    features[to][to_start:to_end] = features[curr][from_start:from_end]

        return features

    def atom_transfer(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Convenience method to transfer atom node features using atom markers.

        Args:
            features: List of atom node feature tensors, modified in-place.

        Returns:
            The updated list of atom node feature tensors.
        """

        return self.transfer_nodes(features, self.markers)

    def bond_transfer(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Convenience method to transfer bond node features using bond markers.

        Requires `use_bond_graph` to be True.

        Args:
            features: List of bond node feature tensors, modified in-place.

        Returns:
            The updated list of bond node feature tensors.
        """
        assert (
            self.use_bond_graph
        ), "Cannot transfer border nodes if use_bond_graph is False"
        return self.transfer_nodes(features, self.line_markers)

    def aggregate_atom_edge(
        self,
        atom_edge_features: List[torch.Tensor],
        gpu_to_aggregate_to: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """
        Aggregates atom graph edge features from all partitions globally.

        Uses `L2G_DE_mapping_list` to map local edge features to their global
        indices in the output tensor.

        Args:
            atom_edge_features: List of local atom edge feature tensors.
            gpu_to_aggregate_to: Target device for the aggregated tensor.
                Defaults to "cpu".

        Returns:
            A single tensor with aggregated edge features, indexed globally.
        """

        aggregated_feats = torch.empty(
            (self.total_num_edges,) + tuple(atom_edge_features[0].shape[1:]),
            device=gpu_to_aggregate_to,
            dtype=atom_edge_features[0].dtype,
        )

        for i, m in enumerate(self.L2G_DE_mapping_list):
            aggregated_feats[m] = atom_edge_features[i].to(gpu_to_aggregate_to)

        return aggregated_feats

    def aggregate_bond_node(
        self,
        bond_node_features: List[torch.Tensor],
        gpu_to_aggregate_to: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """
        Aggregates bond graph node features into an atom-edge aligned tensor.

        Maps bond node features to temporary local atom edge tensors using
        bond-to-edge mappings, then aggregates these temporary tensors globally.
        Edges not represented in the bond graph get zero features.

        Args:
            bond_node_features: List of local bond node feature tensors.
            gpu_to_aggregate_to: Target device for the aggregated tensor.

        Returns:
            Aggregated features tensor, indexed like global atom edges.
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

    def num_atoms(self, partition: int) -> int:
        """
        Returns the total number of atoms in a partition. Includes border nodes.

        Args:
            partition: The partition index.

        Returns:
            Number of local atoms in the partition.
        """
        return len(self.local_coords[partition])

    def num_atom_edges(self, partition):
        """
        Returns the total number of edges within the atom graph of a partition.

        Args:
            partition: The partition index.

        Returns:
            Number of atom graph edges in the partition.
        """
        return len(self.src_nodes[partition])

    def num_bonds(self, partition: int) -> int:
        """
        Returns the total number of local bond graph nodes in a partition.
        Includes border nodes.

        Requires `use_bond_graph` to be True.

        Args:
            partition: The partition index.

        Returns:
            Number of local bond graph nodes in the partition.
        """
        assert self.use_bond_graph, "num_bonds only works when bond graph is enabled"
        return self.line_markers[partition][-1]

    def num_bond_edges(self, partition):
        """
        Returns the number of local bond graph edges in a partition.

        Requires `use_bond_graph` to be True.

        Args:
            partition: The partition index.

        Returns:
            Number of local bond graph edges in the partition.
        """
        assert (
            self.use_bond_graph
        ), "num_bond_edges only works when bond graph is enabled"
        return len(self.line_src_nodes[partition])

    def num_atom_border_nodes(self, partition):
        """
        Returns the number of atom graph border nodes in a partition.

        Args:
            partition: The partition index.

        Returns:
            Number of local atom graph border nodes in the partition.
        """
        return (
            self.num_atoms(partition) - self.markers[partition][1 + self.num_partitions]
        )

    def num_bond_border_nodes(self, partition):
        """
        Returns the number of bond graph border nodes in a partition.

        Requires `use_bond_graph` to be True

        Args:
            partition: The partition index.

        Returns:
            Number of local bond graph border nodes in the partition.
        """
        assert (
            self.use_bond_graph
        ), "num_bond_border_nodes only works when bond graph is enabled"
        return (
            self.num_bonds(partition)
            - self.line_markers[partition][1 + self.num_partitions]
        )

    def global_to_local_nodes(
        self,
        global_node_features: torch.Tensor,
        partition: int,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """
        Extracts local atom graph node features for a partition from a global tensor.

        Uses the `global_ids` mapping to select features.

        Args:
            global_node_features: Tensor with features for all nodes, indexed
                globally.
            partition: Index of the target partition.
            device: Target device for the output local tensor. Defaults to "cpu".

        Returns:
            Tensor with local node features for the specified partition.
        """
        return global_node_features[self.global_ids[partition]].to(device)

    def global_to_local_edges(
        self,
        global_edge_features: torch.Tensor,
        partition: int,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """
        Extracts local atom graph edge features for a partition from a global tensor.

        Uses the `L2G_DE_mapping_list` to select features.

        Args:
            global_edge_features: Tensor with features for all edges, indexed
                globally.
            partition: Index of the target partition.
            device: Target device for the output local tensor. Defaults to "cpu".

        Returns:
            Tensor with local edge features for the specified partition.
        """
        return global_edge_features[self.L2G_DE_mapping_list[partition]].to(device)

    def distribute_node_features(
        self,
        global_node_features: torch.Tensor,
        devices: List[str, torch.device]
    ):
        """
        Distributes global_node_features to the specified devices.

         Args:
            global_node_features: Tensor with features for all nodes, indexed
                globally, of shape [total number of nodes, ...]
            devices: List of the target devices for the local tensors.

        Returns:
            Tensor with local node features for the specified partition.
        """
        assert len(devices) == self.num_partitions, f"There must be the same number of partitions ({self.num_partitions}) as there are devices ({len(devices)})"
        return [self.global_to_local_nodes(global_node_features, i, devices[i]) for i in range(len(devices))]

    def distribute_edge_features(
        self,
        global_edges_features: torch.Tensor,
        devices: List[str, torch.device]
    ):
        """
        Distributes global_edge_features to the specified devices.

         Args:
            global_edge_features: Tensor with features for all edges, indexed
                globally, [total number of edges, ...]
            devices: List of the target devices for the local tensors.

        Returns:
            Tensor with local edge features for the specified partition.
        """
        assert len(devices) == self.num_partitions, f"There must be the same number of partitions ({self.num_partitions}) as there are devices ({len(devices)})"
        return [self.global_to_local_edges(global_edges_features, i, devices[i]) for i in range(len(devices))]

    def edge_to_bond(
        self,
        edge_features: torch.Tensor,
        partition: int,
        device: Union[str, torch.device] = "cpu",
        inplace: bool = False,
        bond_node_features: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Maps atom edge features to bond node features for a partition. This includes
        border nodes within bond graph.

        Uses `bond_mapping_DE_list` and `bond_mapping_UDE_list`. Requires
        `use_bond_graph` to be True.

        Args:
            edge_features: Local atom edge features for the partition.
            partition: The partition index.
            device: Target device for output (if not inplace). Defaults to "cpu".
            inplace: If True, modifies `bond_node_features` directly and
                requires it to be provided. Defaults to False.
            bond_node_features: Pre-allocated tensor for bond node features,
                required and modified if `inplace` is True. Defaults to None.

        Returns:
            New tensor with bond node features if `inplace` is False, else None.
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

    def bond_to_edge(
        self,
        bond_node_features: torch.Tensor,
        atom_edge_features: torch.Tensor,
        partition: int,
    ) -> None:
        """
        Maps bond node features back to atom edge features (in-place).

        Uses `bond_mapping_DE_list` and `bond_mapping_UDE_list`. Modifies
        `atom_edge_features` in-place. Requires `use_bond_graph` to be True.

        Args:
            bond_node_features: Local bond node features for the partition.
            atom_edge_features: Local atom edge features tensor to be modified.
            partition: The partition index.

        Returns:
            None. Modifies `atom_edge_features` in-place.
        """
        atom_edge_features[partition][self.bond_mapping_DE_list[partition]] = (
            bond_node_features[partition][self.bond_mapping_UDE_list[partition]]
        )

    def __repr__(self):
        val = f"""Distributed:
    Total num atoms: {self.total_num_nodes}
    Total num edges: {self.total_num_edges}
    Bond graph exists: {self.use_bond_graph}\n"""

        for i in range(self.num_partitions):
            val += f"Partition {i}:\n"
            val += f"\t# of atom graph nodes: {self.num_atoms(i)} ({self.num_atom_border_nodes(i)} border nodes)\n"
            val += f"\t# of atom graph edges: {len(self.src_nodes[i])}"

            if self.use_bond_graph:
                val += f"\t# of bond graph nodes: {self.num_bonds(i)}. ({self.num_bond_border_nodes(i)} border nodes)\n"
                val += f"\t# of bond graph edges: {self.num_bond_edges(i)}"

            val += "\n"

        return val
