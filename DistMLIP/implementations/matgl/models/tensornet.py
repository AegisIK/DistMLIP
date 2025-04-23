import dgl
import torch
import DistMLIP
from matgl.models import TensorNet
from matgl.utils.maths import decompose_tensor, tensor_norm
from copy import deepcopy


class TensorNet_Dist(TensorNet):
    def potential_forward_dist(
        self,
        dist_info,
        atoms,
        lattice_matrix,
        calc_stresses,
        calc_forces,
        calc_hessian,
        state_attr=None,
    ):
        # strain for stress calculations
        lattice_matrix = torch.tensor(
            lattice_matrix, dtype=self.dtype, device=self.gpus[0]
        )
        strain = lattice_matrix.new_zeros([1, 3, 3], device=self.gpus[0])

        if calc_stresses:
            strain.requires_grad_(True)

        lattice_matrix = lattice_matrix @ (
            torch.eye(3, device=lattice_matrix.device) + strain
        )
        frac_coords = torch.tensor(
            atoms.get_scaled_positions(False), dtype=self.dtype, device=self.gpus[0]
        )

        big_graph_edge_lattice = torch.repeat_interleave(
            lattice_matrix, dist_info.total_num_edges, dim=0
        )
        big_graph_offset = torch.tensor(
            dist_info.py_offsets, dtype=self.dtype, device=self.gpus[0]
        )

        big_graph_offshift = (
            big_graph_offset.unsqueeze(-1) * big_graph_edge_lattice
        ).sum(dim=1)
        big_graph_positions = (
            frac_coords.unsqueeze(dim=-1)
            * torch.repeat_interleave(lattice_matrix, dist_info.total_num_nodes, dim=0)
        ).sum(dim=1)

        if calc_forces:
            big_graph_positions.requires_grad_(True).retain_grad()

        node_types = torch.tensor(
            [self.element_to_index[elem] for elem in atoms.get_chemical_symbols()],
            dtype=DistMLIP.int_th,
            device=self.gpus[0],
        )

        atom_graphs = []
        for partition_i, gpu_index in enumerate(self.gpus):
            # Clone big_graph_positions and index to get the positions we want
            this_gpu_positions = dist_info.global_to_local_nodes(
                big_graph_positions, partition_i, device=gpu_index
            )
            this_gpu_node_types = dist_info.global_to_local_nodes(
                node_types, partition_i, device=gpu_index
            )

            this_gpu_atom_g = dgl.graph(
                (dist_info.src_nodes[partition_i], dist_info.dst_nodes[partition_i]),
                num_nodes=dist_info.num_atoms(partition_i),
            ).to(gpu_index)

            this_gpu_atom_g.ndata["pos"] = this_gpu_positions
            this_gpu_atom_g.ndata["node_type"] = this_gpu_node_types

            atom_graphs.append(this_gpu_atom_g)

        # Calculate all bond_vec and bond_dist for all GPUs on GPU 0
        # In order to calculate this, we need to calculate pbc_offshift for the total graph on GPU 0 as well
        dst_pos = big_graph_positions[dist_info.py_index_2] + big_graph_offshift
        src_pos = big_graph_positions[dist_info.py_index_1]

        big_bond_vec = dst_pos - src_pos
        big_bond_dist = torch.linalg.norm(big_bond_vec, dim=1)

        X_feats = []
        for partition_i, gpu_index in enumerate(self.gpus):
            # Distribute big_bond_vec and big_bond_dist
            atom_graphs[partition_i].edata["bond_vec"] = (
                dist_info.global_to_local_edges(
                    big_bond_vec, partition_i, device=gpu_index
                )
            )
            atom_graphs[partition_i].edata["bond_dist"] = (
                dist_info.global_to_local_edges(
                    big_bond_dist, partition_i, device=gpu_index
                )
            )

            edge_attr = self.bond_expansion_dist[partition_i](
                atom_graphs[partition_i].edata["bond_dist"]
            )
            atom_graphs[partition_i].edata["edge_attr"] = edge_attr

            X, _, _ = self.tensor_embedding_dist[partition_i](
                atom_graphs[partition_i], state_attr
            )
            X_feats.append(X)

        return (
            node_types,
            big_graph_positions,
            strain,
            self.dist_forward(atom_graphs, X_feats, dist_info),
        )

    def dist_forward(self, atom_graphs, X_feats, dist_info):
        # Interaction layers
        for layer_i in range(len(self.layers)):
            for partition_i, gpu_index in enumerate(self.gpus):
                X_feats[partition_i] = self.layers_dist[partition_i][layer_i](
                    atom_graphs[partition_i], X_feats[partition_i]
                )

            # Communicate
            dist_info.atom_transfer(X_feats)

        # Aggregate
        X_aggregated = dist_info.aggregate(X_feats, device=self.gpus[0])

        scalars, skew_metrices, traceless_tensors = decompose_tensor(X_aggregated)
        x = torch.cat(
            (
                tensor_norm(scalars),
                tensor_norm(skew_metrices),
                tensor_norm(traceless_tensors),
            ),
            dim=-1,
        )

        x = self.out_norm_dist(x)
        x = self.linear_dist(x)

        if self.is_intensive:
            raise NotImplementedError(
                "self.is_intensive = True is not yet supported by distributed inference"
            )

        x = self.final_layer_dist.gated(x)

        # Reconstruct the big graph
        big_graph_placeholder = dgl.graph(
            [], num_nodes=dist_info.total_num_nodes, device=self.gpus[0]
        )

        big_graph_placeholder.ndata["atom_features"] = x
        output = dgl.readout_nodes(big_graph_placeholder, "atom_features", op="sum")

        return torch.squeeze(output)

    def enable_distributed_mode(self, gpus):
        if hasattr(self, "dist_enabled") and self.dist_enabled:
            raise Exception("Current model already has distributed mode enabled.")

        # Move everything back to cpu first, the only things on GPU should be the following attributes:
        self.to("cpu")

        self.gpus = []

        for gpu_index in gpus:
            if gpu_index == "cpu":
                self.gpus.append("cpu")
            else:
                self.gpus.append("cuda:" + str(gpu_index))

        self.bond_expansion_dist = [
            deepcopy(self.bond_expansion).to(gpu_index).eval().to(self.dtype)
            for gpu_index in gpus
        ]
        self.tensor_embedding_dist = [
            deepcopy(self.tensor_embedding).to(gpu_index).eval().to(self.dtype)
            for gpu_index in gpus
        ]
        self.layers_dist = [
            [
                deepcopy(layer).to(gpu_index).eval().to(self.dtype)
                for layer in self.layers
            ]
            for gpu_index in gpus
        ]

        self.linear_dist = deepcopy(self.linear).eval().to(self.gpus[0])
        self.final_layer_dist = deepcopy(self.final_layer).eval().to(self.gpus[0])
        self.out_norm_dist = deepcopy(self.out_norm).eval().to(self.gpus[0])

        self.element_to_index = {
            elem: idx for idx, elem in enumerate(self.element_types)
        }
        self.dist_enabled = True

    @classmethod
    def from_existing(cls, model, dtype=DistMLIP.float_th):
        model.to("cpu")
        dist_model = cls.__new__(cls)
        dist_model.__dict__ = model.__dict__.copy()

        dist_model.dist_enabled = False
        dist_model.dtype = dtype

        return dist_model

    def predict_structure_dist(
        self,
        structure,
        state_feats: torch.Tensor | None = None,
    ):
        """Convenience method to directly predict property from structure.

        Args:
            structure: An input crystal/molecule.
            state_feats (torch.tensor): Graph attributes

        Returns:
            output (torch.tensor): output property
        """
        raise NotImplementedError(
            "Distributed direct property prediction is not yet supported. Please raise an issue or use Potential_Dist"
        )
