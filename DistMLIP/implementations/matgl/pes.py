import torch
import numpy as np

from ase import Atoms
from DistMLIP.distributed.dist import Distributed

from matgl.apps.pes import Potential
from matgl.utils.io import IOMixIn

from time import perf_counter


class Potential_Dist(Potential, IOMixIn):
    """A class representing an interatomic potential."""

    __version__ = 2

    def __init__(
        self,
        num_threads = 1,
        **kwargs
    ):
        """
        Initialize Potential from a model and elemental references.

        Args:
            model: Model for predicting energies.
            data_mean: Mean of target.
            data_std: Std dev of target.
            element_refs: Element reference values for each element.
            calc_forces: Enable force calculations.
            calc_stresses: Enable stress calculations.
            calc_hessian: Enable hessian calculations.
            calc_site_wise: Enable site-wise property calculation.
            debug_mode: return gradient of total energy with respect to atomic positions and lattices for checking
        """
        super().__init__(**kwargs)

        assert self.model.dist_enabled, "Distributed mode must be enabled"
        assert hasattr(self.model, "gpus"), "Model should have gpus attribute"
        assert len(self.model.gpus) > 1, "number of GPUs in distributed version should be >1"

        if self.calc_hessian:
            print('Warning: turning off calc_hessian as it is not implemented within distributed inference.')
            self.calc_hessian = False

        self.num_threads = num_threads

    def forward(
        self,
        atoms: Atoms,
        state_attr: torch.Tensor | None = None,
        tol = 1.0e-8,
        num_threads = None,
    ) -> tuple[torch.Tensor, ...]:
        """Args:
            g: DGL graph
            lat: lattice
            state_attr: State attrs
            l_g: Line graph.

        Returns:
            (energies, forces, stresses, hessian) or (energies, forces, stresses, hessian, site-wise properties)
        """
        if not num_threads:
            num_threads = self.num_threads

        ##### Creating Graph Partitions #####
        lattice_matrix = np.array(atoms.get_cell())
        cart_coords = np.array(atoms.get_positions(wrap=False))
        frac_coords = np.array(atoms.get_scaled_positions(wrap=True))

        pbc = np.array([1, 1, 1], dtype=int)
        num_partitions = len(self.model.gpus)

        # TODO: testing, remove when done
        # start = perf_counter()
        
        dist_info = Distributed.create_distributed(cart_coords=cart_coords,
                                                    frac_coords=frac_coords,
                                                    lattice_matrix=lattice_matrix,
                                                    num_partitions=num_partitions,  
                                                    pbc=pbc,
                                                    use_bond_graph=self.model.use_bond_graph if hasattr(self.model, "use_bond_graph") else False,
                                                    cutoff=float(self.model.cutoff),
                                                    three_body_cutoff=float(self.model.three_body_cutoff) if hasattr(self.model, "three_body_cutoff") else 0,
                                                    tol=tol,
                                                    num_threads=num_threads
                                                )
        
        # TODO: testing, remove when done
        # print("Distributed object creation:", perf_counter() - start)

        model_out = self.model.potential_forward_dist(
            dist_info,
            atoms,
            lattice_matrix,
            self.calc_stresses,
            self.calc_forces,
            self.calc_hessian,
            state_attr
        )

        if self.debug_mode:
            print("Debug mode true, returning early")
            return model_out[-1]

        node_types, positions, strain, predictions = model_out

        if isinstance(predictions, tuple) and len(predictions) > 1:
            total_energies, site_wise = predictions
        else:
            total_energies = predictions
            site_wise = None

        total_energies = self.data_std * total_energies + self.data_mean

        if self.element_refs is not None:
            property_offset = torch.squeeze(self.element_refs.atoms_forward_dist(atoms, node_types))
            total_energies += property_offset

        forces = None
        stresses = None
        hessian = None

        grad_vars = [positions, strain] if self.calc_stresses else [positions]

        if self.calc_forces:
            # self.model.zero_grad()
            torch.autograd.backward(total_energies)
            forces = -grad_vars[0].grad


            # grads = grad(
            #     total_energies,
            #     grad_vars,
            #     grad_outputs=torch.ones_like(total_energies),
            #     # create_graph=True,
            #     # retain_graph=True
            # )

            # forces = -grads[0]

        if self.calc_hessian:
            raise NotImplementedError("Calculating hessians is not implemented for distributed inference.")

        if self.calc_stresses:
            volume = np.expand_dims(np.abs(np.linalg.det(lattice_matrix)), 0)
            sts = -grad_vars[1].grad
            scale = 1.0 / volume * -160.21766208
            sts = [i * j for i, j in zip(sts, scale)] if sts.dim() == 3 else [sts * scale]
            stresses = torch.cat(sts)

        return total_energies, forces, stresses, hessian