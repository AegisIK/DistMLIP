from matgl.ext.ase import PESCalculator
from DistMLIP.implementations.matgl.pes import Potential_Dist
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import ExpCellFilter
from ase.filters import FrechetCellFilter
from ase.md import Langevin
from ase.md.andersen import Andersen
from ase.md.bussi import Bussi
from ase.md.npt import NPT
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen, NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet


import numpy as np
import matgl
import torch
from enum import Enum
import sys
import contextlib
import collections
from typing import Literal, Any
import io

from ase.stress import full_3x3_to_voigt_6_stress
from ase.optimize.optimize import Optimizer
import ase.optimize as opt

from pymatgen.core.structure import Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor

from matgl.ext.ase import TrajectoryObserver

# TODO: testing, remove when done
from time import perf_counter

class OPTIMIZERS(Enum):
    """An enumeration of optimizers for used in."""

    fire = opt.fire.FIRE
    bfgs = opt.bfgs.BFGS
    lbfgs = opt.lbfgs.LBFGS
    lbfgslinesearch = opt.lbfgs.LBFGSLineSearch
    mdmin = opt.mdmin.MDMin
    scipyfmincg = opt.sciopt.SciPyFminCG
    scipyfminbfgs = opt.sciopt.SciPyFminBFGS
    bfgslinesearch = opt.bfgslinesearch.BFGSLineSearch


class PESCalculator_Dist(PESCalculator):
    """Machine Learning Interatomic Potential calculator for ASE."""

    implemented_properties = ("energy", "free_energy", "forces", "stress", "hessian", "magmoms")

    def __init__(
        self,
        **kwargs,
    ):
        """
        Init a PESCalculator_Dist with a Potential_Dist.

        Args:
            potential (Potential_Dist): m3gnet.models.Potential_Dist
            state_attr (tensor): State attribute
            compute_stress (bool): whether to calculate the stress
            stress_weight (float): the stress weight.
            **kwargs: Kwargs pass through to super().__init__().
        """
        super().__init__(**kwargs)

        assert isinstance(self.potential, Potential_Dist), \
            ("PESCalculatorDist requires using a Potential_Dist.")
        
        # TODO: testing, remove when done
        self.last_count = None

    def calculate(  # type:ignore[override]
        self,
        atoms: Atoms,
        properties: list | None = None,
        system_changes: list | None = None,
    ):
        """
        Perform calculation for an input Atoms.

        Args:
            atoms (ase.Atoms): ase Atoms object
            properties (list): list of properties to calculate
            system_changes (list): monitor which properties of atoms were
                changed for new calculation. If not, the previous calculation
                results will be loaded.
        """
        properties = properties or ["energy"]
        system_changes = system_changes or all_changes
        Calculator.calculate(self, atoms=atoms, properties=properties, system_changes=system_changes)

        # TODO: testing remove when done
        current_count = perf_counter()
        if self.last_count:
            print("Step time:", current_count - self.last_count)
        self.last_count=current_count
        
        # type: ignore
        if self.state_attr is not None:
            calc_result = self.potential(atoms, self.state_attr)
        else:
            calc_result = self.potential(atoms, np.array([0.0, 0.0]).astype(matgl.float_np))

        self.results.update(
            energy=calc_result[0].detach().cpu().numpy().item(),
            free_energy=calc_result[0].detach().cpu().numpy(),
            forces=calc_result[1].detach().cpu().numpy(),
        )
        if self.compute_stress:
            stresses_np = (
                full_3x3_to_voigt_6_stress(calc_result[2].detach().cpu().numpy())
                if self.use_voigt
                else calc_result[2].detach().cpu().numpy()
            )
            self.results.update(stress=stresses_np * self.stress_weight)
        if self.compute_hessian:
            self.results.update(hessian=calc_result[3].detach().cpu().numpy())
        if self.compute_magmom:
            self.results.update(magmoms=calc_result[4].detach().cpu().numpy())


class Relaxer:
    """Relaxer is a class for structural relaxation."""

    def __init__(
        self,
        potential: Potential_Dist,
        state_attr: torch.Tensor | None = None,
        optimizer: Optimizer | str = "FIRE",
        relax_cell: bool = True,
        stress_weight: float = 1 / 160.21766208,
    ):
        """
        Args:
            potential (Potential): a M3GNet potential, a str path to a saved model or a short name for saved model
            that comes with M3GNet distribution
            state_attr (torch.Tensor): State attr.
            optimizer (str or ase Optimizer): the optimization algorithm.
            Defaults to "FIRE"
            relax_cell (bool): whether to relax the lattice cell
            stress_weight (float): conversion factor from GPa to eV/A^3.
        """
        self.optimizer: Optimizer = OPTIMIZERS[optimizer.lower()].value if isinstance(optimizer, str) else optimizer
        if isinstance(potential, Potential_Dist):
            self.calculator = PESCalculator_Dist(
                potential=potential,
                state_attr=state_attr,
                stress_weight=stress_weight
            )
        else:
            self.calculator = PESCalculator(
                potential=potential,
                state_attr=state_attr,
                stress_weight=stress_weight,  # type: ignore
            )
        self.relax_cell = relax_cell
        self.ase_adaptor = AseAtomsAdaptor()

    def relax(
        self,
        atoms: Atoms | Structure | Molecule,
        fmax: float = 0.1,
        steps: int = 500,
        traj_file: str | None = None,
        interval: int = 1,
        verbose: bool = False,
        ase_cellfilter: Literal["Frechet", "Exp"] = "Frechet",
        params_asecellfilter: dict | None = None,
        **kwargs,
    ):
        """
        Relax an input Atoms.

        Args:
            atoms (Atoms | Structure | Molecule): the atoms for relaxation
            fmax (float): total force tolerance for relaxation convergence.
            Here fmax is a sum of force and stress forces
            steps (int): max number of steps for relaxation
            traj_file (str): the trajectory file for saving
            interval (int): the step interval for saving the trajectories
            verbose (bool): Whether to have verbose output.
            ase_cellfilter (literal): which filter is used for variable cell relaxation. Default is Frechet.
            params_asecellfilter (dict): Parameters to be passed to ExpCellFilter or FrechetCellFilter. Allows
                setting of constant pressure or constant volume relaxations, for example. Refer to
                https://wiki.fysik.dtu.dk/ase/ase/filters.html#FrechetCellFilter for more information.
            **kwargs: Kwargs pass-through to optimizer.
        """
        if isinstance(atoms, Structure | Molecule):
            atoms = self.ase_adaptor.get_atoms(atoms)
        atoms.set_calculator(self.calculator)
        stream = sys.stdout if verbose else io.StringIO()
        params_asecellfilter = params_asecellfilter or {}
        with contextlib.redirect_stdout(stream):
            obs = TrajectoryObserver(atoms)
            if self.relax_cell:
                atoms = (
                    FrechetCellFilter(atoms, **params_asecellfilter)  # type:ignore[assignment]
                    if ase_cellfilter == "Frechet"
                    else ExpCellFilter(atoms, **params_asecellfilter)
                )

            optimizer = self.optimizer(atoms, **kwargs)  # type:ignore[operator]
            optimizer.attach(obs, interval=interval)
            optimizer.run(fmax=fmax, steps=steps)
            obs()
        if traj_file is not None:
            obs.save(traj_file)

        if isinstance(atoms, FrechetCellFilter | ExpCellFilter):
            atoms = atoms.atoms

        return {
            "final_structure": self.ase_adaptor.get_structure(atoms),  # type:ignore[arg-type]
            "trajectory": obs,
        }
    



class MolecularDynamics:
    """Molecular dynamics class."""

    def __init__(
        self,
        atoms: Atoms,
        potential: Potential_Dist,
        state_attr: torch.Tensor | None = None,
        stress_weight: float = 1.0,
        ensemble: Literal[
            "nve", "nvt", "nvt_langevin", "nvt_andersen", "nvt_bussi", "npt", "npt_berendsen", "npt_nose_hoover"
        ] = "nvt",
        temperature: int = 300,
        timestep: float = 1.0,
        pressure: float = 1.01325 * units.bar,
        taut: float | None = None,
        taup: float | None = None,
        friction: float = 1.0e-3,
        andersen_prob: float = 1.0e-2,
        ttime: float = 25.0,
        pfactor: float = 75.0**2.0,
        external_stress: float | np.ndarray | None = None,
        compressibility_au: float | None = None,
        trajectory: Any = None,
        logfile: str | None = None,
        loginterval: int = 1,
        append_trajectory: bool = False,
        mask: tuple | np.ndarray | None = None,
    ):
        """
        Init the MD simulation.

        Args:
            atoms (Atoms): atoms to run the MD
            potential (Potential): potential for calculating the energy, force,
            stress of the atoms
            state_attr (torch.Tensor): State attr.
            stress_weight (float): conversion factor from GPa to eV/A^3
            ensemble (str): choose from "nve", "nvt", "nvt_langevin", "nvt_andersen", "nvt_bussi",
            "npt", "npt_berendsen", "npt_nose_hoover"
            temperature (float): temperature for MD simulation, in K
            timestep (float): time step in fs
            pressure (float): pressure in eV/A^3
            taut (float): time constant for Berendsen temperature coupling
            taup (float): time constant for pressure coupling
            friction (float): friction coefficient for nvt_langevin, typically set to 1e-4 to 1e-2
            andersen_prob (float): random collision probability for nvt_andersen, typically set to 1e-4 to 1e-1
            ttime (float): Characteristic timescale of the thermostat, in ASE internal units
            pfactor (float): A constant in the barostat differential equation.
            external_stress (float): The external stress in eV/A^3.
                Either 3x3 tensor,6-vector or a scalar representing pressure
            compressibility_au (float): compressibility of the material in A^3/eV
            trajectory (str or Trajectory): Attach trajectory object
            logfile (str): open this file for recording MD outputs
            loginterval (int): write to log file every interval steps
            append_trajectory (bool): Whether to append to prev trajectory.
            mask (np.array): either a tuple of 3 numbers (0 or 1) or a symmetric 3x3 array indicating,
                which strain values may change for NPT simulations.
        """
        if isinstance(atoms, Structure | Molecule):
            atoms = AseAtomsAdaptor().get_atoms(atoms)
        self.atoms = atoms

        if isinstance(potential, Potential_Dist):
            self.atoms.set_calculator(
                PESCalculator_Dist(potential=potential, state_attr=state_attr, stress_unit="eV/A3", stress_weight=stress_weight)
            )
        else:
            self.atoms.set_calculator(
                PESCalculator(potential=potential, state_attr=state_attr, stress_unit="eV/A3", stress_weight=stress_weight)
            )

        if taut is None:
            taut = 100 * timestep * units.fs
        if taup is None:
            taup = 1000 * timestep * units.fs
        if mask is None:
            mask = np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
        if external_stress is None:
            external_stress = 0.0

        if ensemble.lower() == "nvt":
            self.dyn = NVTBerendsen(
                self.atoms,
                timestep * units.fs,
                temperature_K=temperature,
                taut=taut,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                append_trajectory=append_trajectory,
            )

        elif ensemble.lower() == "nve":
            self.dyn = VelocityVerlet(  # type:ignore[assignment]
                self.atoms,
                timestep * units.fs,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                append_trajectory=append_trajectory,
            )

        elif ensemble.lower() == "nvt_langevin":
            self.dyn = Langevin(  # type:ignore[assignment]
                self.atoms,
                timestep * units.fs,
                temperature_K=temperature,
                friction=friction,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                append_trajectory=append_trajectory,
            )

        elif ensemble.lower() == "nvt_andersen":
            self.dyn = Andersen(  # type:ignore[assignment]
                self.atoms,
                timestep * units.fs,
                temperature_K=temperature,
                andersen_prob=andersen_prob,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                append_trajectory=append_trajectory,
            )

        elif ensemble.lower() == "nvt_bussi":
            if np.isclose(self.atoms.get_kinetic_energy(), 0.0, rtol=0, atol=1e-12):
                MaxwellBoltzmannDistribution(self.atoms, temperature_K=temperature)
            self.dyn = Bussi(  # type:ignore[assignment]
                self.atoms,
                timestep * units.fs,
                temperature_K=temperature,
                taut=taut,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                append_trajectory=append_trajectory,
            )

        elif ensemble.lower() == "npt":
            """
            NPT ensemble default to Inhomogeneous_NPTBerendsen thermo/barostat
            This is a more flexible scheme that fixes three angles of the unit
            cell but allows three lattice parameter to change independently.
            """

            self.dyn = Inhomogeneous_NPTBerendsen(  # type:ignore[assignment]
                self.atoms,
                timestep * units.fs,
                temperature_K=temperature,
                pressure_au=pressure,
                taut=taut,
                taup=taup,
                compressibility_au=compressibility_au,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                append_trajectory=append_trajectory,
            )

        elif ensemble.lower() == "npt_berendsen":
            """

            This is a similar scheme to the Inhomogeneous_NPTBerendsen.
            This is a less flexible scheme that fixes the shape of the
            cell - three angles are fixed and the ratios between the three
            lattice constants.

            """

            self.dyn = NPTBerendsen(
                self.atoms,
                timestep * units.fs,
                temperature_K=temperature,
                pressure_au=pressure,
                taut=taut,
                taup=taup,
                compressibility_au=compressibility_au,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                append_trajectory=append_trajectory,
            )

        elif ensemble.lower() == "npt_nose_hoover":
            self.upper_triangular_cell()
            self.dyn = NPT(  # type:ignore[assignment]
                self.atoms,
                timestep * units.fs,
                temperature_K=temperature,
                externalstress=external_stress,  # type:ignore[arg-type]
                ttime=ttime * units.fs,
                pfactor=pfactor * units.fs,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                append_trajectory=append_trajectory,
                mask=mask,
            )

        else:
            raise ValueError("Ensemble not supported")

        self.trajectory = trajectory
        self.logfile = logfile
        self.loginterval = loginterval
        self.timestep = timestep

    def run(self, steps: int):
        """Thin wrapper of ase MD run.

        Args:
            steps (int): number of MD steps
        """
        self.dyn.run(steps)

    def set_atoms(self, atoms: Atoms):
        """Set new atoms to run MD.

        Args:
            atoms (Atoms): new atoms for running MD.
        """
        if isinstance(atoms, Structure | Molecule):
            atoms = AseAtomsAdaptor().get_atoms(atoms)
        calculator = self.atoms.calc
        self.atoms = atoms
        self.dyn.atoms = atoms
        self.dyn.atoms.set_calculator(calculator)

    def upper_triangular_cell(self, verbose: bool | None = False) -> None:
        """Transform to upper-triangular cell.
        ASE Nose-Hoover implementation only supports upper-triangular cell
        while ASE's canonical description is lower-triangular cell.

        Args:
            verbose (bool): Whether to notify user about upper-triangular cell
                transformation. Default = False
        """
        if not NPT._isuppertriangular(self.atoms.get_cell()):
            a, b, c, alpha, beta, gamma = self.atoms.cell.cellpar()
            angles = np.radians((alpha, beta, gamma))
            sin_a, sin_b, _sin_g = np.sin(angles)
            cos_a, cos_b, cos_g = np.cos(angles)
            cos_p = (cos_g - cos_a * cos_b) / (sin_a * sin_b)
            cos_p = np.clip(cos_p, -1, 1)
            sin_p = (1 - cos_p**2) ** 0.5

            new_basis = [
                (a * sin_b * sin_p, a * sin_b * cos_p, a * cos_b),
                (0, b * sin_a, b * cos_a),
                (0, 0, c),
            ]

            self.atoms.set_cell(new_basis, scale_atoms=True)
            if verbose:
                print("Transformed to upper triangular unit cell.", flush=True)
