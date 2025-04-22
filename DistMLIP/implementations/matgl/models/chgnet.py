from matgl.models import CHGNet
import torch
from dgl import readout_edges, readout_nodes
from matgl.utils.cutoff import polynomial_cutoff

import numpy as np

from copy import deepcopy
import matgl

from matgl.graph.compute import (
    compute_pair_vector_and_distance,
    compute_theta,
    create_line_graph,
    ensure_line_graph_compatibility,
)

import dgl

class CHGNet_Dist(CHGNet):
    """Main CHGNet model."""

    __version__ = 1

    def potential_forward_dist(
            self,
            dist_info,
            atoms,
            lattice_matrix,
            calc_stresses,
            calc_forces,
            calc_hessian,
            state_attr=None
        ):
        ##### Creating 1 Global tensor for positions (on self.gpus[0]) and distributing to each features for each subgraph #####
        # strain for stress calculations
        lattice_matrix = torch.tensor(lattice_matrix, dtype=matgl.dist_float_th, device=self.gpus[0])
        strain = lattice_matrix.new_zeros([1, 3, 3], device = self.gpus[0])

        if calc_stresses:
            strain.requires_grad_(True)

        lattice_matrix = lattice_matrix @ (torch.eye(3, device=lattice_matrix.device) + strain)
        frac_coords = torch.tensor(atoms.get_scaled_positions(False), dtype=matgl.dist_float_th, device=self.gpus[0])


        big_graph_edge_lattice = torch.repeat_interleave(lattice_matrix, dist_info.total_num_edges, dim=0)
        big_graph_offset = torch.tensor(dist_info.py_offsets, dtype=matgl.dist_float_th, device=self.gpus[0])

        big_graph_offshift = (big_graph_offset.unsqueeze(-1) * big_graph_edge_lattice).sum(dim=1)
        big_graph_positions = (frac_coords.unsqueeze(dim=-1) * torch.repeat_interleave(lattice_matrix, dist_info.total_num_nodes, dim=0)).sum(dim=1)

        if calc_forces:
            big_graph_positions.requires_grad_(True).retain_grad()
    
        node_types = torch.tensor(np.array([self.element_to_index[elem] for elem in atoms.get_chemical_symbols()]), dtype=matgl.int_th, device=self.gpus[0])

        atom_graphs = []
        for partition_i, gpu_index in enumerate(self.gpus):
            # Clone big_graph_positions and index to get the positions we want
            this_gpu_positions = dist_info.global_to_local_nodes(big_graph_positions, partition_i, device=gpu_index)
            this_gpu_node_types = dist_info.global_to_local_nodes(node_types, partition_i, device=gpu_index)

            this_gpu_atom_g = dgl.graph((dist_info.src_nodes[partition_i], dist_info.dst_nodes[partition_i]), num_nodes=dist_info.num_atoms(partition_i)).to(gpu_index)

            this_gpu_atom_g.ndata["pos"] = this_gpu_positions
            this_gpu_atom_g.ndata["node_type"] = this_gpu_node_types
            
            atom_graphs.append(this_gpu_atom_g)
        
        # Calculate all bond_vec and bond_dist for all GPUs on GPU 0
        # In order to calculate this, we need to calculate pbc_offshift for the total graph on GPU 0 as well
        dst_pos = big_graph_positions[dist_info.py_index_2] + big_graph_offshift
        src_pos = big_graph_positions[dist_info.py_index_1]

    
        big_bond_vec = (dst_pos - src_pos)
        big_bond_dist = torch.linalg.norm(big_bond_vec, dim=1)

        for partition_i, gpu_index in enumerate(self.gpus):
            # Distribute big_bond_vec and big_bond_dist
            atom_graphs[partition_i].edata["bond_vec"] = dist_info.global_to_local_edges(big_bond_vec, partition_i, device=gpu_index)
            atom_graphs[partition_i].edata["bond_dist"] = dist_info.global_to_local_edges(big_bond_dist, partition_i, device=gpu_index)
            

            # Element-wise expansion + polynomial cutoff for edata in atom graph
            bond_expansion = self.bond_expansion_dist[partition_i](atom_graphs[partition_i].edata["bond_dist"])
            smooth_cutoff = polynomial_cutoff(bond_expansion, self.cutoff, self.cutoff_exponent)
            atom_graphs[partition_i].edata["bond_expansion"] = smooth_cutoff * bond_expansion

        # Create bond graphs
        bond_graphs = []

        if self.use_bond_graph:
            # First, set up bond_dist and bond_vec attributes in bond graph node features
            for partition_i, gpu_index in enumerate(self.gpus):
                num_nodes = dist_info.num_bonds(partition_i)
                this_bond_graph = dgl.graph((dist_info.line_src_nodes[partition_i], dist_info.line_dst_nodes[partition_i]), num_nodes = num_nodes).to(gpu_index)

                # Send bond_dist data from edges in atom_graph to nodes in this_bond_graph
                this_bond_graph.ndata['bond_dist'] = dist_info.edge_to_bond(atom_graphs[partition_i].edata["bond_dist"], partition_i, device=gpu_index)

                # Send bond_vec data from edges in atom_graph to nodes in this bond_graph
                this_bond_graph.ndata['bond_vec'] = dist_info.edge_to_bond(atom_graphs[partition_i].edata["bond_vec"], partition_i, device=gpu_index)

                bond_graphs.append(this_bond_graph)

            # Perform data transfer in order to receive border bond_dist and bond_vec information as well

            dist_info.bond_transfer([bond_graph.ndata['bond_dist'] for bond_graph in bond_graphs])
            dist_info.bond_transfer([bond_graph.ndata['bond_vec'] for bond_graph in bond_graphs])

            for partition_i, gpu_index in enumerate(self.gpus):
                this_bond_graph = bond_graphs[partition_i]
                num_nodes = dist_info.num_bonds(partition_i)

                # Perform RBF on bond_dist data
                threebody_bond_expansion = self.threebody_bond_expansion_dist[partition_i](this_bond_graph.ndata["bond_dist"])
                smooth_cutoff = polynomial_cutoff(threebody_bond_expansion, self.three_body_cutoff, self.cutoff_exponent)

                this_bond_graph.ndata["bond_expansion"] = smooth_cutoff * threebody_bond_expansion

                # Set up src_bond_sign and bond_index
                this_bond_graph.edata["center_atom_index"] = torch.tensor(dist_info.local_center_atom_indices_list[partition_i], device=gpu_index, dtype=torch.int64)
                this_bond_graph.ndata["src_bond_sign"] = -1 * torch.ones(num_nodes, 1, device=gpu_index)

                this_bond_graph.apply_edges(compute_theta)
                this_bond_graph.edata["angle_expansion"] = self.angle_expansion_dist[partition_i](this_bond_graph.edata["theta"])
    
        return node_types, big_graph_positions, strain, self.dist_forward(atom_graphs=atom_graphs, bond_graphs=bond_graphs, dist_info=dist_info)


    def dist_forward(self,
        atom_graphs: [dgl.DGLGraph],
        bond_graphs: [dgl.DGLGraph],
        dist_info,
        state_features: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:

        num_partitions = dist_info.num_partitions
        # Compute state, atom, bond and angle embeddings
        atom_features = []
        atom_edge_features = [] 
        bond_node_features = [] # NOTE: this shares many entries as atom_edge features. 2 differences: 
                                # 1. bond graph nodes don't have all of atom graph's edge features
                                # 2. atom graph edges don't include bond graph border nodes
        angle_features = []

        if not state_features is None:
            raise NotImplementedError("State features not implemented for distributed computation.")

        state_features = None

        for partition_i in range(num_partitions):
            curr_atom_graph = atom_graphs[partition_i]
            
            atom_features.append(self.atom_embedding_dist[partition_i](curr_atom_graph.ndata["node_type"]))
            atom_edge_features.append(self.bond_embedding_dist[partition_i](curr_atom_graph.edata["bond_expansion"]))

            if self.use_bond_graph:
                curr_bond_graph = bond_graphs[partition_i]
                angle_features.append(self.angle_embedding_dist[partition_i](curr_bond_graph.edata["angle_expansion"]))

        if self.use_bond_graph:
            # Construct the bond_node_features tensors
            for partition_i, gpu_index in enumerate(self.gpus):
                # NOTE: technically, this line is not necessary. Commenting out but keeping for readability's sake TODO: comment this out later
                this_bond_node_feature = dist_info.edge_to_bond(atom_edge_features[partition_i], partition_i, device=gpu_index)

                bond_node_features.append(this_bond_node_feature)
            
            # Perform inter-GPU message passing in order to obtain the rest of the necessary bond_node_features
            dist_info.bond_transfer(bond_node_features)

        # Setting up shared message weights
        atom_bond_weights = [None] * len(self.gpus)
        bond_bond_weights = [None] * len(self.gpus)
        threebody_bond_weights = [None] * len(self.gpus)

        # Shared message weights
        if self.atom_bond_weights:
            atom_bond_weights = [self.atom_bond_weights_dist[partition_i](atom_graphs[partition_i].edata["bond_expansion"]) for partition_i in range(len(self.gpus))]

        if self.bond_bond_weights:
            bond_bond_weights = [self.bond_bond_weights_dist[partition_i](atom_graphs[partition_i].edata["bond_expansion"]) for partition_i in range(len(self.gpus))]

        if self.threebody_bond_weights:
            threebody_bond_weights = [self.threebody_bond_weights_dist[partition_i](bond_graphs[partition_i].ndata["bond_expansion"]) for partition_i in range(len(self.gpus))]

        if self.use_bond_graph:
            # Pass messages
            for layer_i in range(self.n_blocks - 1):
                for partition_i, gpu_index in enumerate(self.gpus):
                    atom_features[partition_i], atom_edge_features[partition_i], _ = self.atom_graph_layers_dist[partition_i][layer_i]( 
                        atom_graphs[partition_i], atom_features[partition_i], atom_edge_features[partition_i], None, atom_bond_weights[partition_i], bond_bond_weights[partition_i]
                    )

                    # Update bond_node_features with newly computed atom_edge_features
                    dist_info.edge_to_bond(atom_edge_features, partition_i, inplace=True, bond_node_features=bond_node_features)

                dist_info.bond_transfer(bond_node_features)
                dist_info.atom_transfer(atom_features)

                for partition_i, gpu_index in enumerate(self.gpus):
                    # Take bond features and extract the necessary bond graph node features, concatenate additional information from other GPUs, then pass into this function w/ compute_dist enabled
                    # Convolve the bond graph
                    bond_node_features[partition_i], angle_features[partition_i] = self.bond_graph_layers_dist[partition_i][layer_i](bond_graphs[partition_i], atom_features[partition_i], bond_node_features[partition_i], angle_features[partition_i], threebody_bond_weights[partition_i], compute_dist=True, convolution_type="node")

                    # Update the atom_edge_features with the new output
                    dist_info.bond_to_edge(bond_node_features, atom_edge_features, partition_i)

                dist_info.bond_transfer(bond_node_features)
                
                # Update the angle features (edges in bond graph)
                for partition_i, gpu_index in enumerate(self.gpus):
                    bond_node_features[partition_i], angle_features[partition_i] = self.bond_graph_layers_dist[partition_i][layer_i](bond_graphs[partition_i], atom_features[partition_i], bond_node_features[partition_i], angle_features[partition_i], threebody_bond_weights[partition_i], compute_dist=True, convolution_type="edge")

        else:
            # Pass messages
            for layer_i in range(self.n_blocks - 1):
                for partition_i, gpu_index in enumerate(self.gpus):

                    atom_features[partition_i], atom_edge_features[partition_i], _ = self.atom_graph_layers_dist[partition_i][layer_i]( 
                        atom_graphs[partition_i], atom_features[partition_i], atom_edge_features[partition_i], None, atom_bond_weights[partition_i], bond_bond_weights[partition_i]
                    )

                dist_info.atom_transfer(atom_features)

        # Site-wise target readout (per partition, and then re-aggregate)
        site_properties_list = [self.sitewise_readout_dist[partition_i](atom_features[partition_i]) for partition_i in range(num_partitions)]
        site_properties_aggregate = dist_info.aggregate(site_properties_list, self.gpus[0])
        
        # Last atom graph message passing layer -------------------------------
        layer_i = -1
        for partition_i, gpu_index in enumerate(self.gpus):
            atom_features[partition_i], atom_edge_features[partition_i], _ = self.atom_graph_layers_dist[partition_i][layer_i](
                atom_graphs[partition_i], atom_features[partition_i], atom_edge_features[partition_i], None, atom_bond_weights[partition_i], bond_bond_weights[partition_i]
            )

        # Pass messages for atom and bond graph between GPUs
        dist_info.atom_transfer(atom_features)

        # ---------------------------------
        if self.readout_field == "atom_feat":
            # Perform final layer pass on all atom features
            final_atom_features = [self.final_layer_dist[partition_i](atom_features[partition_i]) for partition_i in range(num_partitions)]
            
            # Reconstruct the big graph
            big_graph_placeholder = dgl.graph([], num_nodes = dist_info.total_num_nodes, device = self.gpus[0])
            combined_atom_feats = dist_info.aggregate(final_atom_features, self.gpus[0])

            big_graph_placeholder.ndata["atom_features"] = combined_atom_feats
            structure_properties_aggregate = readout_nodes(big_graph_placeholder, "atom_features", op = self.readout_operation)

            return structure_properties_aggregate, site_properties_aggregate

        elif self.readout_field == "bond_feat":
            raise NotImplementedError("bond_feat readout not yet supported. Let me (Kevin/AegisIK) know if you need this.")
        elif self.readout_field == "angle_feat":
            raise NotImplementedError("angle_feat readout not yet supported. Let me (Kevin/AegisIK) know if you need this.")
        else:
            raise Exception(f"Unknown self.readout_field value of: {self.readout_field}")

    def enable_distributed_mode(self, gpus):
        """Should only be done when you wish to make forward passes in distributed mode"""
        if self.dist_enabled:
            raise Exception("Current model already has distributed mode enabled.")

        # Move everything back to cpu first, the only things on GPU should be the following attributes:
        self.to("cpu")


        self.gpus = []

        for gpu_index in gpus:
            if gpu_index == "cpu":
                self.gpus.append("cpu")
            else:
                self.gpus.append("cuda:" + str(gpu_index))

        self.bond_expansion_dist = [deepcopy(self.bond_expansion).to(gpu_index).eval() for gpu_index in gpus]

        self.threebody_bond_expansion_dist = [deepcopy(self.threebody_bond_expansion).to(gpu_index).eval() for gpu_index in gpus] if self.use_bond_graph else None
        self.angle_expansion_dist = [deepcopy(self.angle_expansion).to(gpu_index).eval() for gpu_index in gpus] if self.use_bond_graph else None

        if self.atom_bond_weights:
            self.atom_bond_weights_dist = [deepcopy(self.atom_bond_weights).to(gpu_index).eval() for gpu_index in gpus]
        
        if self.bond_bond_weights:
            self.bond_bond_weights_dist = [deepcopy(self.bond_bond_weights).to(gpu_index).eval() for gpu_index in gpus]
        
        if self.threebody_bond_weights:
            self.threebody_bond_weights_dist = [deepcopy(self.threebody_bond_weights).to(gpu_index).eval() for gpu_index in gpus]
        
        self.atom_graph_layers_dist = [deepcopy(self.atom_graph_layers).to(gpu_index).eval() for gpu_index in gpus]
        self.bond_graph_layers_dist = [deepcopy(self.bond_graph_layers).to(gpu_index).eval() for gpu_index in gpus] if self.use_bond_graph else None

        self.atom_embedding_dist = [deepcopy(self.atom_embedding).to(gpu_index).eval() for gpu_index in gpus]
        self.bond_embedding_dist = [deepcopy(self.bond_embedding).to(gpu_index).eval() for gpu_index in gpus]
        self.angle_embedding_dist = [deepcopy(self.angle_embedding).to(gpu_index).eval() for gpu_index in gpus] if self.use_bond_graph else None

        if not self.state_embedding is None:
            self.state_embedding_dist = [deepcopy(self.state_embedding).to(gpu_index).eval() for gpu_index in gpus]

        self.sitewise_readout_dist = [deepcopy(self.sitewise_readout).to(gpu_index).eval() for gpu_index in gpus]


        self.final_layer_dist = [deepcopy(self.final_layer).to(gpu_index).eval() for gpu_index in gpus]

        self.element_to_index = {elem: idx for idx, elem in enumerate(self.element_types)}
        self.dist_enabled = True

    @classmethod
    def from_existing(cls, model):
        dist_model = cls.__new__(cls)
        dist_model.__dict__ = model.__dict__.copy()

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
        raise NotImplementedError("Distributed direct property prediction is not yet supported. Please raise an issue or use Potential_Dist")