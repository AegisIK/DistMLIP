from mace.modules.models import ScaleShiftMACE
from copy import deepcopy
import DistMLIP

from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode
from mace.tools import torch_tools

from DistMLIP.distributed.dist import Distributed
from DistMLIP.implementations.mace.mace_utils import prepare_graph
from copy import deepcopy

from mace.modules.radial import ZBLBasis
from mace.tools.scatter import scatter_sum

from mace.modules.blocks import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearDipoleReadoutBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearDipoleReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    ScaleShiftBlock,
)
from mace.modules.utils import (
    compute_fixed_charge_dipole,
    get_atomic_virials_stresses,
    get_edge_vectors_and_lengths,
    get_outputs,
    get_symmetric_displacement,
)


class ScaleShiftMACE_Dist(ScaleShiftMACE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def dist_forward(
        self,
        data: Dict[str, torch.Tensor],
        dist_info,
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
        compute_edge_forces: bool = False,
        compute_atomic_stresses: bool = False,
        lammps_mliap: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        ctx = prepare_graph(
            data,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
            lammps_mliap=lammps_mliap,
        )

        is_lammps = ctx.is_lammps
        num_atoms_arange = ctx.num_atoms_arange
        num_graphs = ctx.num_graphs
        displacement = ctx.displacement
        positions = ctx.positions
        vectors = ctx.vectors
        lengths = ctx.lengths
        cell = ctx.cell
        node_heads = ctx.node_heads
        interaction_kwargs = ctx.interaction_kwargs
        lammps_natoms = interaction_kwargs.lammps_natoms
        lammps_class = interaction_kwargs.lammps_class

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])[
            num_atoms_arange, node_heads
        ]

        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=0, dim_size=num_graphs
        )  # [n_graphs, num_heads]

        node_attrs_dist = dist_info.distribute_node_features(data["node_attrs"], self.gpus)
        node_heads_dist = dist_info.distribute_node_features(node_heads, self.gpus)

        vectors_dist = dist_info.distribute_edge_features(vectors, self.gpus)
        lengths_dist = dist_info.distribute_edge_features(lengths, self.gpus)

        src_nodes_torch = [torch.from_numpy(dist_info.src_nodes[i]) for i in range(len(self.gpus))]
        dst_nodes_torch = [torch.from_numpy(dist_info.dst_nodes[i]) for i in range(len(self.gpus))]
        edge_index_dist = [torch.cat((src[None, :], dst[None, :]), dim=0).to(gpu) for gpu, src, dst in zip(self.gpus, src_nodes_torch, dst_nodes_torch)]

        # Embeddings distributed
        node_feats_dist = []
        edge_attrs_dist = []
        edge_feats_dist = []
        

        # Embeddings
        for partition_i, gpu_index in enumerate(self.gpus):
            curr_node_feats = self.node_embedding_dist[partition_i](node_attrs_dist[partition_i])
            curr_edge_attrs = self.spherical_harmonics_dist[partition_i](vectors_dist[partition_i])

            curr_edge_feats = self.radial_embedding_dist[partition_i](lengths_dist[partition_i], 
                                                                      node_attrs_dist[partition_i], 
                                                                      edge_index_dist[partition_i],
                                                                      self.atomic_numbers_dist[partition_i]
                                                                      )

            node_feats_dist.append(curr_node_feats)
            edge_attrs_dist.append(curr_edge_attrs)
            edge_feats_dist.append(curr_edge_feats)

        if hasattr(self, "pair_repulsion"):
            pair_node_energy_dist = []
            for partition_i, gpu_index in enumerate(self.gpus):
                pair_node_energy_dist.append(self.pair_repulsion_fn_dist[partition_i](lengths_dist[partition_i], node_attrs_dist[partition_i], edge_index_dist[partition_i], self.atomic_numbers_dist[partition_i]))

            pair_node_energy = dist_info.aggregate(pair_node_energy_dist, self.gpus[0])
        else:
            pair_node_energy = torch.zeros_like(node_e0)

        # Interactions
        node_es_list = [pair_node_energy]
        node_feats_list: List[torch.Tensor] = []
        
        node_es_dist = [None] * len(self.gpus)

        for i in range(len(self.interactions)):
            for partition_i, gpu_index in enumerate(self.gpus):
                curr_node_attrs_slice = node_attrs_dist[partition_i]
                curr_node_feats = node_feats_dist[partition_i]
                curr_edge_attrs = edge_attrs_dist[partition_i]
                curr_edge_feats = edge_feats_dist[partition_i]
                curr_edge_index = edge_index_dist[partition_i]

                curr_node_feats, sc = self.interactions_dist[partition_i][i](
                    node_attrs=curr_node_attrs_slice,
                    node_feats=curr_node_feats,
                    edge_attrs=curr_edge_attrs,
                    edge_feats=curr_edge_feats,
                    edge_index=curr_edge_index,
                    first_layer=(i == 0),
                    lammps_class=lammps_class,
                    lammps_natoms=lammps_natoms,
                )

                curr_node_feats = self.products_dist[partition_i][i](
                    node_feats=curr_node_feats, sc=sc, node_attrs=curr_node_attrs_slice
                )

                readout_result = self.readouts_dist[partition_i][i](curr_node_feats, node_heads_dist[partition_i])[torch.arange(dist_info.num_atoms(partition_i), device=gpu_index), node_heads_dist[partition_i]]
                
                node_feats_dist[partition_i] = curr_node_feats
                node_es_dist[partition_i] = readout_result

            # Transfer information for node_feats_dist
            dist_info.atom_transfer(node_feats_dist)

            # Aggregate information for node_feats_dist and append to node_feats_list
            node_feats_list.append(dist_info.aggregate(node_feats_dist, self.gpus[0]))

            # Aggregate information for node_es_dist and append to node_es_list
            node_es_list.append(dist_info.aggregate(node_es_dist, self.gpus[0]))

        node_feats_out = torch.cat(node_feats_list, dim=-1)
        node_inter_es = torch.sum(torch.stack(node_es_list, dim=0), dim=0)
        node_inter_es = self.scale_shift(node_inter_es, node_heads)
        inter_e = scatter_sum(node_inter_es, data["batch"], dim=-1, dim_size=num_graphs)

        total_energy = e0 + inter_e
        node_energy = node_e0.double() + node_inter_es.double()

        forces, virials, stress, hessian, edge_forces = get_outputs(
            energy=inter_e,
            positions=positions,
            displacement=displacement,
            vectors=vectors,
            cell=cell,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_hessian=compute_hessian,
            compute_edge_forces=compute_edge_forces,
        )

        atomic_virials: Optional[torch.Tensor] = None
        atomic_stresses: Optional[torch.Tensor] = None
        if compute_atomic_stresses and edge_forces is not None:
            atomic_virials, atomic_stresses = get_atomic_virials_stresses(
                edge_forces=edge_forces,
                edge_index=data["edge_index"],
                vectors=vectors,
                num_atoms=positions.shape[0],
                batch=data["batch"],
                cell=cell,
            )

        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "interaction_energy": inter_e,
            "forces": forces,
            "edge_forces": edge_forces,
            "virials": virials,
            "stress": stress,
            "atomic_virials": atomic_virials,
            "atomic_stresses": atomic_stresses,
            "hessian": hessian,
            "displacement": displacement,
            "node_feats": node_feats_out,
        }

    def enable_distributed_mode(self, gpus):
        self.gpus = []

        for gpu_index in gpus:
            if gpu_index == "cpu":
                self.gpus.append("cpu")
            else:
                self.gpus.append("cuda:" + str(gpu_index))
        # self.to("cpu")
        # self.device = torch_tools.init_device(self.gpus[0])

        self.atomic_energies_fn_dist = [deepcopy(self.atomic_energies_fn).to(gpu_index) for gpu_index in self.gpus]

        self.node_embedding_dist = [deepcopy(self.node_embedding).to(gpu_index) for gpu_index in self.gpus]
        self.spherical_harmonics_dist = [deepcopy(self.spherical_harmonics).to(gpu_index) for gpu_index in self.gpus]
        self.radial_embedding_dist = [deepcopy(self.radial_embedding).to(gpu_index) for gpu_index in self.gpus]

        self.atomic_numbers_dist = [deepcopy(self.atomic_numbers).to(gpu_index) for gpu_index in self.gpus]

        self.interactions_dist = [deepcopy(self.interactions).to(gpu_index) for gpu_index in self.gpus]
        self.products_dist = [deepcopy(self.products).to(gpu_index) for gpu_index in self.gpus]
        self.readouts_dist = [deepcopy(self.readouts).to(gpu_index) for gpu_index in self.gpus]

        self.scale_shift_dist = deepcopy(self.scale_shift).to(self.gpus[0])

        if hasattr(self, "pair_repulsion"):
            self.pair_repulsion_fn_dist = [deepcopy(self.pair_repulsion_fn).to(gpu_index) for gpu_index in self.gpus]

        self.dist_enabled = True

    @classmethod
    def from_existing(cls, model):
        if not isinstance(model, ScaleShiftMACE):
            raise NotImplementedError(f"Only ScaleShiftMACE supports distributed mode at the moment, not {type(model)}. Contact Kevin at kevinhan@cmu.edu if you want to request a new model to be supported.")

        model.to("cpu")
        dist_model = cls.__new__(cls)
        dist_model.__dict__ = model.__dict__.copy()

        dist_model.dist_enabled = False

        return dist_model
