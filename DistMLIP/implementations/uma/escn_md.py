"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.nn as nn
from torch.profiler import record_function

from fairchem.core.common import gp_utils
from fairchem.core.common.distutils import get_device_for_local_rank
from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad
from DistMLIP.implementations.uma.compute import generate_graph
from fairchem.core.models.base import HeadInterface
from fairchem.core.models.uma.common.rotation import (
    init_edge_rot_mat,
    rotation_to_wigner,
)
from fairchem.core.models.uma.common.rotation_cuda_graph import RotMatWignerCudaGraph
from fairchem.core.models.uma.common.so3 import CoefficientMapping, SO3_Grid
from fairchem.core.models.uma.nn.embedding_dev import (
    ChgSpinEmbedding,
    DatasetEmbedding,
    EdgeDegreeEmbedding,
)
from fairchem.core.models.uma.nn.layer_norm import (
    EquivariantLayerNormArray,
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    get_normalization_layer,
)
from fairchem.core.models.uma.nn.mole_utils import MOLEInterface
from fairchem.core.models.uma.nn.radial import GaussianSmearing
from fairchem.core.models.uma.nn.so3_layers import SO3_Linear
from fairchem.core.models.utils.irreps import cg_change_mat, irreps_sum

from fairchem.core.models.uma.escn_md_block import eSCNMD_Block
from fairchem.core.models.uma.escn_md import eSCNMDBackbone

import functools
import threading
import torch
from copy import deepcopy
from fairchem.core.models.uma.nn.mole_utils import (
    model_search_and_replace,
    replace_linear_with_MOLE,
    recursive_replace_so2_linear,
    recursive_replace_all_linear,
    recursive_replace_notso2_linear,
    recursive_replace_so2_MOLE
)

from concurrent.futures import ThreadPoolExecutor
import threading
import types

ESCNMD_DEFAULT_EDGE_CHUNK_SIZE = 1024 * 128


def dist_set_coefficients(existing_module, expert_mixing_coefficients, mole_sizes):
    existing_module.global_mole_tensors.expert_mixing_coefficients = expert_mixing_coefficients
    existing_module.global_mole_tensors.mole_sizes = mole_sizes
    return existing_module

def _get_rotmat_and_wigner(
        self, edge_distance_vecs: torch.Tensor, use_cuda_graph: bool, partition:int=None
    ):
        Jd_buffers = [
            getattr(self, f"Jd_{l}").type(edge_distance_vecs.dtype)
            for l in range(self.lmax + 1)
        ]

        if partition:
            Jd_buffers = [
                self.Jd_list_dist[(l, partition)].type(edge_distance_vecs.dtype)
                for l in range(self.lmax + 1)
            ]

        if use_cuda_graph:
            if self.rot_mat_wigner_cuda is None:
                self.rot_mat_wigner_cuda = RotMatWignerCudaGraph()
            with record_function("obtain rotmat wigner cudagraph"):
                edge_rot_mat, wigner, wigner_inv = (
                    self.rot_mat_wigner_cuda.get_rotmat_and_wigner(
                        edge_distance_vecs, Jd_buffers
                    )
                )
        else:
            with record_function("obtain rotmat wigner original"):
                edge_rot_mat = init_edge_rot_mat(
                    edge_distance_vecs, rot_clip=(not self.direct_forces)
                )
                wigner = rotation_to_wigner(
                    edge_rot_mat,
                    0,
                    self.lmax,
                    Jd_buffers,
                    rot_clip=(not self.direct_forces),
                )
                wigner_inv = torch.transpose(wigner, 1, 2).contiguous()

        # select subset of coefficients we are using
        if self.mmax != self.lmax:
            wigner = wigner.index_select(1, self.coefficient_index.to(Jd_buffers[0].device))
            wigner_inv = wigner_inv.index_select(2, self.coefficient_index.to(Jd_buffers[0].device))

        if partition:
            wigner_and_M_mapping = torch.einsum(
                "mk,nkj->nmj", self.mappingReduced_dist[partition].to_m.to(wigner.dtype), wigner
            )
            wigner_and_M_mapping_inv = torch.einsum(
                "njk,mk->njm", wigner_inv, self.mappingReduced_dist[partition].to_m.to(wigner_inv.dtype)
            )
        else:
            wigner_and_M_mapping = torch.einsum(
                "mk,nkj->nmj", self.mappingReduced.to_m.to(wigner.dtype), wigner
            )
            wigner_and_M_mapping_inv = torch.einsum(
                "njk,mk->njm", wigner_inv, self.mappingReduced.to_m.to(wigner_inv.dtype)
            )
        return edge_rot_mat, wigner_and_M_mapping, wigner_and_M_mapping_inv


def _generate_graph(self, data_dict, num_partitions=1):
    if self.otf_graph:
        pbc = None
        if self.always_use_pbc:
            pbc = torch.ones(len(data_dict), 3, dtype=torch.bool)
        else:
            assert (
                "pbc" in data_dict
            ), "Since always_use_pbc is False, pbc conditions must be supplied by the input data"
            pbc = data_dict["pbc"]
        assert (
            pbc.all() or (~pbc).all()
        ), "We can only accept pbc that is all true or all false"
        logging.debug(f"Using radius graph gen version {self.radius_pbc_version}")
        graph_dict = generate_graph(
            data_dict,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
            enforce_max_neighbors_strictly=self.enforce_max_neighbors_strictly,
            radius_pbc_version=self.radius_pbc_version,
            pbc=pbc,
            num_partitions=num_partitions
        )
    else:
        # this assume edge_index is provided
        assert (
            "edge_index" in data_dict
        ), "otf_graph is false, need to provide edge_index as input!"
        cell_per_edge = data_dict["cell"].repeat_interleave(
            data_dict["nedges"], dim=0
        )
        shifts = torch.einsum(
            "ij,ijk->ik",
            data_dict["cell_offsets"].to(cell_per_edge.dtype),
            cell_per_edge,
        )
        edge_distance_vec = (
            data_dict["pos"][data_dict["edge_index"][0]]
            - data_dict["pos"][data_dict["edge_index"][1]]
            + shifts
        )  # [n_edges, 3]
        # pylint: disable=E1102
        edge_distance = torch.linalg.norm(
            edge_distance_vec, dim=-1, keepdim=False
        )  # [n_edges, 1]

        graph_dict = {
            "edge_index": data_dict["edge_index"],
            "edge_distance": edge_distance,
            "edge_distance_vec": edge_distance_vec,
            "node_offset": 0,
        }

    if gp_utils.initialized():
        graph_dict = self._init_gp_partitions(
            graph_dict, data_dict["atomic_numbers_full"]
        )
        # create partial atomic numbers and batch tensors for GP
        node_partition = graph_dict["node_partition"]
        data_dict["atomic_numbers"] = data_dict["atomic_numbers_full"][
            node_partition
        ]
        data_dict["batch"] = data_dict["batch_full"][node_partition]
    else:
        graph_dict["node_offset"] = 0
        graph_dict["edge_distance_vec_full"] = graph_dict["edge_distance_vec"]
        graph_dict["edge_distance_full"] = graph_dict["edge_distance"]
        graph_dict["edge_index_full"] = graph_dict["edge_index"]

    return graph_dict

def _part_forward_worker(self, part_idx, module, args, kwargs):
    """
    Runs one partition's module forward on its GPU.
    Returns (output, finish_event).
    """
    dev = self.gpus[part_idx]
    torch.cuda.set_device(dev)

    # Use default stream by default; this matches typical PyTorch behavior and parallel_apply.
    # If you want per-thread custom streams, see "Advanced stream control" notes below.
    out = module(*args, **kwargs)

    # Record an event that marks the completion of this partition's enqueued work on the default stream.
    ev = torch.cuda.Event(blocking=False, enable_timing=False)
    torch.cuda.current_stream(dev).record_event(ev)
    return out, ev

def _init_edge_embed(self, part_idx, edge_distance_part, atomic_num_part, edge_index_part, x_message_part, wigner_inv_part):
    edge_distance_embedding_part = self.distance_expansion_dist[part_idx](
        edge_distance_part
    )

    source_embedding_part = self.source_embedding_dist[part_idx](
        atomic_num_part[edge_index_part[0]]
    )

    target_embedding_part = self.target_embedding_dist[part_idx](
        atomic_num_part[edge_index_part[1]]
    )

    x_edge_part = torch.cat(
        (edge_distance_embedding_part, source_embedding_part, target_embedding_part), dim=1
    )

    x_message_part = self.edge_degree_embedding_dist[part_idx](
        x_message_part,
        x_edge_part,
        edge_distance_part,
        edge_index_part,
        wigner_inv_part,
        0
    )

    return x_message_part, x_edge_part

@conditional_grad(torch.enable_grad())
def forward(self, data_dict) -> dict[str, torch.Tensor]:
    data_dict["atomic_numbers"] = data_dict["atomic_numbers"].long()
    data_dict["atomic_numbers_full"] = data_dict["atomic_numbers"]
    data_dict["batch_full"] = data_dict["batch"]

    csd_mixed_emb = self.csd_embedding(
        charge=data_dict["charge"],
        spin=data_dict["spin"],
        dataset=data_dict.get("dataset", None),
    )

    self.set_MOLE_coefficients(
        atomic_numbers_full=data_dict["atomic_numbers_full"],
        batch_full=data_dict["batch_full"],
        csd_mixed_emb=csd_mixed_emb,
    )

    with record_function("get_displacement_and_cell"):
        displacement, orig_cell = self._get_displacement_and_cell(data_dict)

    with record_function("generate_graph"):
        if hasattr(self, "dist_enabled") and self.dist_enabled:
            graph_dict = self._generate_graph(data_dict, num_partitions=len(self.gpus))
        else:
            graph_dict = self._generate_graph(data_dict, num_partitions=1)

    if graph_dict["edge_index"].numel() == 0:
        raise ValueError(
            f"No edges found in input system, this means either you have a single atom in the system or the atoms are farther apart than the radius cutoff of the model of {self.cutoff} Angstroms. We don't know how to handle this case. Check the positions of system: {data_dict['pos']}"
        )

    dist_info = graph_dict["dist_info"]

    with record_function("obtain wigner"):
        if hasattr(self, "dist_enabled") and self.dist_enabled:
            edge_distance_vec_full_dist = dist_info.distribute_edge_features(graph_dict["edge_distance_vec_full"], self.gpus)

            wigner_and_M_mapping_dist = [None] * len(self.gpus)
            wigner_and_M_mapping_inv_dist = [None] * len(self.gpus)
            # Run synchronously first
            for partition_i in range(len(self.gpus)):
                _, wigner_and_M_mapping_dist[partition_i], wigner_and_M_mapping_inv_dist[partition_i] = self._get_rotmat_and_wigner(edge_distance_vec_full_dist[partition_i], partition=partition_i, use_cuda_graph=False)
        else:
            (edge_rot_mat, wigner_and_M_mapping_full, wigner_and_M_mapping_inv_full) = (
                self._get_rotmat_and_wigner(
                    graph_dict["edge_distance_vec_full"],
                    use_cuda_graph=self.use_cuda_graph_wigner
                    and "cuda" in get_device_for_local_rank()
                    and not self.training,
                )
            )
            # As a sanity check this should all be 0, dist, 0 (dist = scalar distance)
            # rotated_ones = torch.bmm(edge_rot_mat, graph_dict["edge_distance_vec"].unsqueeze(-1)).squeeze(-1)
            if gp_utils.initialized():
                wigner_and_M_mapping = wigner_and_M_mapping_full[
                    graph_dict["edge_partition"]
                ]
                wigner_and_M_mapping_inv = wigner_and_M_mapping_inv_full[
                    graph_dict["edge_partition"]
                ]
            else:
                wigner_and_M_mapping = wigner_and_M_mapping_full
                wigner_and_M_mapping_inv = wigner_and_M_mapping_inv_full

    ###############################################################
    # Initialize node embeddings
    ###############################################################

    # Init per node representations using an atomic number based embedding
    with record_function("atom embedding"):
        x_message = torch.zeros(
            data_dict["atomic_numbers"].shape[0],
            self.sph_feature_size,
            self.sphere_channels,
            device=data_dict["pos"].device,
            dtype=data_dict["pos"].dtype,
        )
        x_message[:, 0, :] = self.sphere_embedding(data_dict["atomic_numbers"])

    sys_node_embedding = csd_mixed_emb[data_dict["batch"]]
    x_message[:, 0, :] = x_message[:, 0, :] + sys_node_embedding

    ###
    # Hook to allow MOLE
    ###
    self.set_MOLE_sizes(
        nsystems=csd_mixed_emb.shape[0],
        batch_full=data_dict["batch_full"],
        edge_index=graph_dict["edge_index"],
    )
    self.log_MOLE_stats()

    # For distributed mode, distribute both expert_mixing_coefficients and mole_sizes
    if hasattr(self, "dist_enabled") and self.dist_enabled:
        with record_function("distribute coefficients"):
            # Need to send the global_mole_tensors.expert_mixing_coefficients to the proper gpu and block_dist
            for partition_i in range(len(self.gpus)):
                curr_expert_mixing_coefficients = self.global_mole_tensors.expert_mixing_coefficients.to(self.gpus[partition_i])
                curr_mole_sizes=self.global_mole_tensors.mole_sizes.to(self.gpus[partition_i])
                for blocks in self.blocks_dist[partition_i]:

                    replace = functools.partial(
                        dist_set_coefficients,
                        expert_mixing_coefficients=curr_expert_mixing_coefficients,
                        mole_sizes=curr_mole_sizes
                    )

                    recursive_replace_so2_MOLE(blocks, replace)#, target_device=self.gpus[partition_i])

        with record_function("distribute features"):
            x_message_dist = dist_info.distribute_node_features(x_message, self.gpus)
            sys_node_embedding_dist = dist_info.distribute_node_features(sys_node_embedding, self.gpus)
            # wigner_and_M_mapping_dist = dist_info.distribute_edge_features(wigner_and_M_mapping, self.gpus)
            # wigner_and_M_mapping_inv_dist = dist_info.distribute_edge_features(wigner_and_M_mapping_inv, self.gpus)
            edge_distance_dist = dist_info.distribute_edge_features(graph_dict["edge_distance"], self.gpus)
            atomic_numbers_dist = dist_info.distribute_node_features(data_dict["atomic_numbers_full"], self.gpus)
            x_edge_dist = [None] * len(self.gpus) # calculated later

            # Build edge_index per partition; pinned host -> async H2D (as you already do)
            src_nodes_host = [torch.from_numpy(dist_info.src_nodes[i]).pin_memory() for i in range(len(self.gpus))]
            dst_nodes_host = [torch.from_numpy(dist_info.dst_nodes[i]).pin_memory() for i in range(len(self.gpus))]
            edge_index_dist = []
            for gpu, src_h, dst_h in zip(self.gpus, src_nodes_host, dst_nodes_host):
                src = src_h.to(gpu, non_blocking=True)
                dst = dst_h.to(gpu, non_blocking=True)
                edge_index_dist.append(torch.stack((src, dst), dim=0))

    # edge degree embedding
    with record_function("edge embedding"):
        if hasattr(self, "dist_enabled") and self.dist_enabled:
            # Submit the job to workers
            inputs = [
                (
                    p, edge_distance_dist[p], atomic_numbers_dist[p], 
                    edge_index_dist[p], x_message_dist[p], wigner_and_M_mapping_inv_dist[p]
                ) for p in range(len(self.gpus))
            ]

            futures = [
                self._executor.submit(lambda part_idx, args, kwargs: self._part_forward_worker(part_idx, self._init_edge_embed, args, kwargs),
                                        p, inputs[p], {})
                for p in range(len(self.gpus))
            ]

            # Collect results (same order)
            outs = []
            first_exc = None
            for f in futures:
                try:
                    outs.append(f.result())
                except Exception as e:
                    # Propagate first exception while making a best-effort to cancel others
                    first_exc = first_exc or e
            if first_exc is not None:
                for f in futures:
                    f.cancel()
                raise first_exc

            # Slot outputs and make default streams wait on the worker events
            for p, (out, ev) in enumerate(outs):
                x_message_dist[p], x_edge_dist[p] = out
                # Attach happens-after to the default stream on this device
                ds = torch.cuda.default_stream(self.gpus[p])
                ds.wait_event(ev)

            dist_info.atom_transfer(x_message_dist)

        else:
            edge_distance_embedding = self.distance_expansion(
                graph_dict["edge_distance"]
            )
            source_embedding = self.source_embedding(
                data_dict["atomic_numbers_full"][graph_dict["edge_index"][0]]
            )
            target_embedding = self.target_embedding(
                data_dict["atomic_numbers_full"][graph_dict["edge_index"][1]]
            )
            x_edge = torch.cat(
                (edge_distance_embedding, source_embedding, target_embedding), dim=1
            )
            x_message = self.edge_degree_embedding(
                x_message,
                x_edge,
                graph_dict["edge_distance"],
                graph_dict["edge_index"],
                wigner_and_M_mapping_inv,
                graph_dict["node_offset"],
            )

    ###############################################################
    # Update spherical node embeddings
    ###############################################################
    if hasattr(self, "dist_enabled") and self.dist_enabled:
        with self._fw_lock:
            num_parts = len(self.gpus)

            # Parallel per-layer forward across partitions
            for i in range(self.num_layers):
                with record_function(f"message passing {i}"):
                    modules = [self.blocks_dist[p][i] for p in range(num_parts)]

                    inputs = [
                        (
                            x_message_dist[p],
                            x_edge_dist[p],
                            edge_distance_dist[p],
                            edge_index_dist[p],
                            wigner_and_M_mapping_dist[p],
                            wigner_and_M_mapping_inv_dist[p],
                        )
                        for p in range(num_parts)
                    ]
                    kwargs_list = [
                        {"sys_node_embedding": sys_node_embedding_dist[p], "node_offset": 0}
                        for p in range(num_parts)
                    ]

                    # Submit per-partition jobs
                    futures = [
                        self._executor.submit(self._part_forward_worker, p, modules[p], inputs[p], kwargs_list[p])
                        for p in range(num_parts)
                    ]

                    # Collect results (same order)
                    outs = []
                    first_exc = None
                    for f in futures:
                        try:
                            outs.append(f.result())
                        except Exception as e:
                            # Propagate first exception while making a best-effort to cancel others
                            first_exc = first_exc or e
                    if first_exc is not None:
                        for f in futures:
                            f.cancel()
                        raise first_exc

                    # Slot outputs and make default streams wait on the worker events
                    for p, (out, ev) in enumerate(outs):
                        x_message_dist[p] = out
                        # Attach happens-after to the default stream on this device
                        ds = torch.cuda.default_stream(self.gpus[p])
                        ds.wait_event(ev)

                    # Inter-partition halo exchange
                    # This will enqueue its work after the per-partition compute on the default streams
                    dist_info.atom_transfer(x_message_dist)

            # Aggregate then final norm on a chosen device (GPU 0 here)
            x_message = dist_info.aggregate(x_message_dist, device=self.gpus[0])
            x_message = self.norm(x_message)
    else:
        for i in range(self.num_layers):
            with record_function(f"message passing {i}"):
                x_message = self.blocks[i](
                    x_message, # num nodes
                    x_edge, # edges
                    graph_dict["edge_distance"],
                    graph_dict["edge_index"],
                    wigner_and_M_mapping, # edges
                    wigner_and_M_mapping_inv, # edges
                    sys_node_embedding=sys_node_embedding, # nodes
                    node_offset=graph_dict["node_offset"], # scalar (0)
                )

        # Final layer norm
        x_message = self.norm(x_message)
    out = {
        "node_embedding": x_message,
        "displacement": displacement,
        "orig_cell": orig_cell,
        "batch": data_dict["batch"],
    }
    return out

def enable_distributed_mode(self, gpus):
    if hasattr(self, "dist_enabled"):
        assert not self.dist_enabled, "Distributed mode already enabled. Create a new UMA model if you wish to change the GPUs."
    
    self.gpus = []

    for gpu_index in gpus:
        if gpu_index == "cpu":
            self.gpus.append("cpu")
        else:
            self.gpus.append("cuda:" + str(gpu_index))
    
    self.blocks_dist = [deepcopy(self.blocks).to(gpu_index) for gpu_index in self.gpus]
    self.distance_expansion_dist = [deepcopy(self.distance_expansion).to(gpu_index) for gpu_index in self.gpus]
    self.source_embedding_dist = [deepcopy(self.source_embedding).to(gpu_index) for gpu_index in self.gpus]
    self.target_embedding_dist = [deepcopy(self.target_embedding).to(gpu_index) for gpu_index in self.gpus]
    self.edge_degree_embedding_dist = [deepcopy(self.edge_degree_embedding).to(gpu_index) for gpu_index in self.gpus]

    # Distribute Jd_list
    Jd_list = torch.load(os.path.join(os.path.dirname(__file__), "Jd.pt"))
    self.Jd_list_dist = {}
    for partition_i, gpu in enumerate(self.gpus):
        for l in range(self.lmax + 1):
            self.Jd_list_dist[(l, partition_i)] = Jd_list[l].clone().to(gpu)

    self.mappingReduced_dist = [deepcopy(self.mappingReduced).to(gpu_index) for gpu_index in self.gpus]

    # Create workers
    self._executor = ThreadPoolExecutor(max_workers=len(self.gpus), thread_name_prefix="part")
    self._fw_lock = threading.RLock()

    self.dist_enabled = True

# Attach the necessary functions to the model
def from_existing(model):
    if not isinstance(model, eSCNMDBackbone):
        raise NotImplementedError(f"Only eSCNMDBackbone and child models are supported for distributed mode at the moment, not {type(model)}. Contact Kevin at kevinhan@cmu.edu if you want to request a new model to be supported.")
    
    model.forward = types.MethodType(forward, model)
    model.enable_distributed_mode = types.MethodType(enable_distributed_mode, model)
    model._init_edge_embed = types.MethodType(_init_edge_embed, model)
    model._part_forward_worker = types.MethodType(_part_forward_worker, model)
    model._generate_graph = types.MethodType(_generate_graph, model)
    model._get_rotmat_and_wigner = types.MethodType(_get_rotmat_and_wigner, model)

    return model