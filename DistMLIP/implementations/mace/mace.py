from ase.calculators.calculator import Calculator, all_changes
from mace.calculators.mace import MACECalculator
from mace.tools import torch_geometric, torch_tools, utils
from mace.modules.utils import extract_invariant

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from e3nn import o3
from mace import data

from DistMLIP.implementations.mace.mace_utils import AtomicData_Dist
from DistMLIP.implementations.mace.models import ScaleShiftMACE_Dist

class MACECalculator_Dist(MACECalculator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _atoms_to_batch(self, atoms, dist_inference=False):
        if dist_inference:
            assert hasattr(self, "dist_enabled") and self.dist_enabled, "You need to call enable_distributed_mode before performing distributed inference"
        
        keyspec = data.KeySpecification(
            info_keys={}, arrays_keys={"charges": self.charges_key}
        )
        config = data.config_from_atoms(
            atoms, key_specification=keyspec, head_name=self.head
        )

        if dist_inference:
            atomic_data, dist_info = AtomicData_Dist.from_config(
                    config,
                    z_table=self.z_table,
                    cutoff=self.r_max,
                    heads=self.available_heads,
                    dist_inference=True,
                    num_partitions=len(self.gpus)
                )
        else:
            atomic_data = data.AtomicData.from_config(
                        config,
                        z_table=self.z_table,
                        cutoff=self.r_max,
                        heads=self.available_heads,
                    )
            dist_info = None

        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                atomic_data
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        batch = next(iter(data_loader)).to(self.device)
        return batch, dist_info
    def calculate(self, atoms=None, properties=None, system_changes=all_changes, dist_inference=True, debug=False):
        """
        Calculate properties.
        :param atoms: ase.Atoms object
        :param properties: [str], properties to be computed, used by ASE internally
        :param system_changes: [str], system changes since last calculation, used by ASE internally
        :return:
        """
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)
        if dist_inference:
            assert hasattr(self, "dist_enabled") and self.dist_enabled, "Must call enable_distributed_mode before running distributed inference"

        batch_base, dist_info = self._atoms_to_batch(atoms, dist_inference)

        if self.model_type in ["MACE", "EnergyDipoleMACE"]:
            batch = self._clone_batch(batch_base)
            node_heads = batch["head"][batch["batch"]]
            num_atoms_arange = torch.arange(batch["positions"].shape[0])
            node_e0 = self.models[0].atomic_energies_fn(batch["node_attrs"])[
                num_atoms_arange, node_heads
            ]
            compute_stress = not self.use_compile
        else:
            compute_stress = False

        ret_tensors = self._create_result_tensors(
            self.model_type, self.num_models, len(atoms)
        )
        for i, model in enumerate(self.models):
            batch = self._clone_batch(batch_base)

            if dist_inference:
                out = model.dist_forward(
                    batch.to_dict(),
                    compute_stress=compute_stress,
                    training=self.use_compile,
                    dist_info=dist_info)

                if debug:
                    return out
            else:
                out = model(
                    batch.to_dict(),
                    compute_stress=compute_stress,
                    training=self.use_compile,
                )

                if debug:
                    return out
            if self.model_type in ["MACE", "EnergyDipoleMACE"]:
                ret_tensors["energies"][i] = out["energy"].detach()
                ret_tensors["node_energy"][i] = (out["node_energy"] - node_e0).detach()
                ret_tensors["forces"][i] = out["forces"].detach()
                if out["stress"] is not None:
                    ret_tensors["stress"][i] = out["stress"].detach()
            if self.model_type in ["DipoleMACE", "EnergyDipoleMACE"]:
                ret_tensors["dipole"][i] = out["dipole"].detach()

        self.results = {}
        if self.model_type in ["MACE", "EnergyDipoleMACE"]:
            self.results["energy"] = (
                torch.mean(ret_tensors["energies"], dim=0).cpu().item()
                * self.energy_units_to_eV
            )
            self.results["free_energy"] = self.results["energy"]
            self.results["node_energy"] = (
                torch.mean(ret_tensors["node_energy"], dim=0).cpu().numpy()
            )
            self.results["forces"] = (
                torch.mean(ret_tensors["forces"], dim=0).cpu().numpy()
                * self.energy_units_to_eV
                / self.length_units_to_A
            )
            if self.num_models > 1:
                self.results["energies"] = (
                    ret_tensors["energies"].cpu().numpy() * self.energy_units_to_eV
                )
                self.results["energy_var"] = (
                    torch.var(ret_tensors["energies"], dim=0, unbiased=False)
                    .cpu()
                    .item()
                    * self.energy_units_to_eV
                )
                self.results["forces_comm"] = (
                    ret_tensors["forces"].cpu().numpy()
                    * self.energy_units_to_eV
                    / self.length_units_to_A
                )
            if out["stress"] is not None:
                self.results["stress"] = full_3x3_to_voigt_6_stress(
                    torch.mean(ret_tensors["stress"], dim=0).cpu().numpy()
                    * self.energy_units_to_eV
                    / self.length_units_to_A**3
                )
                if self.num_models > 1:
                    self.results["stress_var"] = full_3x3_to_voigt_6_stress(
                        torch.var(ret_tensors["stress"], dim=0, unbiased=False)
                        .cpu()
                        .numpy()
                        * self.energy_units_to_eV
                        / self.length_units_to_A**3
                    )
        if self.model_type in ["DipoleMACE", "EnergyDipoleMACE"]:
            self.results["dipole"] = (
                torch.mean(ret_tensors["dipole"], dim=0).cpu().numpy()
            )
            if self.num_models > 1:
                self.results["dipole_var"] = (
                    torch.var(ret_tensors["dipole"], dim=0, unbiased=False)
                    .cpu()
                    .numpy()
                )

    @classmethod
    def from_existing(cls, existing):
        # Call from_existing on all models
        models = [ScaleShiftMACE_Dist.from_existing(model) for model in existing.models]
        
        dist_calc = cls.__new__(cls)
        dist_calc.__dict__ = existing.__dict__.copy()
        
        dist_calc.dist_enabled = False
        dist_calc.models = models
        return dist_calc

    def enable_distributed_mode(self, gpus):
        for model in self.models:
            model.to(gpus[0])
            
            model.enable_distributed_mode(gpus)

        self.device = torch_tools.init_device(self.models[0].gpus[0])
        self.gpus = gpus

        self.dist_enabled = True

