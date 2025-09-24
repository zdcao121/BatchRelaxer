# -*- coding: utf-8 -*-
# https://github.com/microsoft/mattersim/blob/main/src/mattersim/applications/batch_relax.py
import sys
import numpy as np
from typing import Dict, List, Union

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import Filter
from ase.filters import ExpCellFilter, FrechetCellFilter
from ase.optimize import BFGS, FIRE
from ase.optimize.optimize import Optimizer
from loguru import logger
from tqdm import tqdm

from ase.build import bulk
from orb_models.forcefield import atomic_system, pretrained
from orb_models.forcefield.base import batch_graphs


# adapted from https://github.com/orbital-materials/orb-models/blob/6847b1c01386f6d4b8a4c78929185b6cb0c75cf9/orb_models/forcefield/calculator.py
class DummyBatchCalculator(Calculator):
    def __init__(self):
        super().__init__()

    def calculate(self, atoms=None, properties=None, system_changes=None):
        pass

    def get_potential_energy(self, atoms=None):
        raw_energy = atoms.info["total_energy"]
        atoms.info["total_energy"] = float(raw_energy)
        return atoms.info["total_energy"]

    def get_forces(self, atoms=None):
        return atoms.arrays["forces"]

    def get_stress(self, atoms=None):
        raw_stress = atoms.info["stress"]
        atoms.info["stress"] = (
                raw_stress[0] if len(raw_stress.shape) > 1 else raw_stress
            )
        return atoms.info["stress"]


class BatchRelaxer(object):
    """BatchRelaxer is a class for batch structural relaxation.
    It is more efficient than Relaxer when relaxing a large number of structures."""

    SUPPORTED_OPTIMIZERS = {"BFGS": BFGS, "FIRE": FIRE}
    SUPPORTED_FILTERS = {
        "EXPCELLFILTER": ExpCellFilter,
        "FRECHETCELLFILTER": FrechetCellFilter,
    }

    def __init__(
        self,
        potential,
        device: str = "cuda",
        optimizer: Union[str, type[Optimizer]] = "FIRE",
        filter: Union[type[Filter], str, None] = None,
        fmax: float = 0.05,
        max_natoms_per_batch: int = 512,
        max_n_steps: int = 1_000_000,
    ):
        self.potential = potential
        self.device = device
        if isinstance(optimizer, str):
            if optimizer.upper() not in self.SUPPORTED_OPTIMIZERS:
                raise ValueError(f"Unsupported optimizer: {optimizer}")
            self.optimizer = self.SUPPORTED_OPTIMIZERS[optimizer.upper()]
        elif issubclass(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        if isinstance(filter, str):
            if filter.upper() not in self.SUPPORTED_FILTERS:
                raise ValueError(f"Unsupported filter: {filter}")
            self.filter = self.SUPPORTED_FILTERS[filter.upper()]
        elif filter is None or issubclass(filter, Filter):
            self.filter = filter
        else:
            raise ValueError(f"Unsupported filter: {filter}")
        self.fmax = fmax
        self.max_natoms_per_batch = max_natoms_per_batch
        self.optimizer_instances: List[Optimizer] = []
        self.is_active_instance: List[bool] = []
        self.finished = False
        self.total_converged = 0
        self.trajectories: Dict[int, List[Atoms]] = {}
        self.max_n_steps = max_n_steps 

    def insert(self, atoms: Atoms):
        atoms.calc = DummyBatchCalculator()
        optimizer_instance = self.optimizer(
            self.filter(atoms) if self.filter else atoms
        )
        optimizer_instance.fmax = self.fmax
        optimizer_instance.nsteps = 0
        self.optimizer_instances.append(optimizer_instance)
        self.is_active_instance.append(True)

    def step_batch(self):
        atoms_list = []
        for idx, opt in enumerate(self.optimizer_instances):
            if self.is_active_instance[idx]:
                atoms_list.append(opt.atoms)

        # Note: we use a batch size of len(atoms_list)
        # because we only want to run one batch at a time
        ####################### Modified for ORB models #######################
        graph_batch = batch_graphs([atomic_system.ase_atoms_to_atom_graphs(ii.atoms, self.potential.system_config, device=self.device) for ii in atoms_list])    
        result = self.potential.predict(graph_batch)

        energy_batch = result['energy']
        forces_batch = result.get("forces", result.get("grad_forces"))
        stress_batch = result.get("stress", result.get("grad_stress"))
        energy_batch = energy_batch.detach().cpu().numpy()
        forces_batch = forces_batch.detach().cpu().numpy()
        stress_batch = stress_batch.detach().cpu().numpy()

        atom_lengths = np.array([len(ii.atoms) for ii in atoms_list])
        cumulative_lengths = np.cumsum(atom_lengths)  # cumulative lengths
        # split the forces into the individual atoms
        forces_batch = np.split(forces_batch, cumulative_lengths[:-1])
        ######################################################################

        counter = 0
        self.finished = True
        for idx, opt in enumerate(self.optimizer_instances):
            if self.is_active_instance[idx]:
                # Set the properties so the dummy calculator can
                # return them within the optimizer step
                opt.atoms.info["total_energy"] = energy_batch[counter]
                opt.atoms.arrays["forces"] = forces_batch[counter]
                opt.atoms.info["stress"] = stress_batch[counter]
                try:
                    self.trajectories[opt.atoms.info["structure_index"]].append(
                        opt.atoms.copy()
                    )
                except KeyError:
                    self.trajectories[opt.atoms.info["structure_index"]] = [
                        opt.atoms.copy()
                    ]

                opt.step()
                opt.nsteps += 1
                if opt.converged() or opt.nsteps >= self.max_n_steps:
                    self.is_active_instance[idx] = False
                    self.total_converged += 1
                    if self.total_converged % 100 == 0:
                        logger.info(f"Relaxed {self.total_converged} structures.")
                else:
                    self.finished = False
                counter += 1

        # remove inactive instances
        self.optimizer_instances = [
            opt
            for opt, active in zip(self.optimizer_instances, self.is_active_instance)
            if active
        ]
        self.is_active_instance = [True] * len(self.optimizer_instances)

    def relax(
        self,
        atoms_list: List[Atoms],
    ) -> Dict[int, List[Atoms]]:
        self.trajectories = {}
        self.tqdmcounter = tqdm(total=len(atoms_list), file=sys.stdout)
        pointer = 0
        atoms_list_ = []
        for i in range(len(atoms_list)):
            atoms_list_.append(atoms_list[i].copy())
            atoms_list_[i].info["structure_index"] = i

        while (
            pointer < len(atoms_list) or not self.finished
        ):  # While there are unfinished instances or atoms left to insert
            while pointer < len(atoms_list) and (
                sum([len(opt.atoms) for opt in self.optimizer_instances])
                + len(atoms_list[pointer])
                <= self.max_natoms_per_batch
            ):
                # While there are enough n_atoms slots in the
                # batch and we have not reached the end of the list.
                self.insert(
                    atoms_list_[pointer]
                )  # Insert new structure to fire instances
                self.tqdmcounter.update(1)
                pointer += 1
            self.step_batch()
        self.tqdmcounter.close()

        return self.trajectories


if __name__ == "__main__":
    from ase.build import bulk
    from time import time

    device = "cuda"  # or device="cuda"
    potential = pretrained.orb_v2("./data/orb-v2-20241011.ckpt", device=device)
    
    # initialize the batch relaxer with a EXPCELLFILTER for cell relaxation and a FIRE optimizer
    relaxer = BatchRelaxer(potential,
                           device=device,
                           fmax=0.01,
                           filter="FRECHETCELLFILTER",
                           optimizer="FIRE")

    # Here, we generate a list of ASE Atoms objects we want to relax
    atoms_list = [bulk("C"), bulk("Mg"), bulk("Si"), bulk("Ni")]

    # And then perturb them a bit so that relaxation is not trivial
    for atoms in atoms_list:
        atoms.rattle(stdev=0.1)

    # Run the relaxation
    s_time = time()
    relaxation_trajectories = relaxer.relax(atoms_list)
    e_time = time()
    print(f"Time: {e_time-s_time}")

    # Extract the relaxed structures and corresponding energies
    relaxed_structures = [traj[-1] for traj in relaxation_trajectories.values()]
    relaxed_energies = [structure.info['total_energy'] for structure in relaxed_structures]

    # Do the same with the initial structures and energies
    initial_structures = [traj[0] for traj in relaxation_trajectories.values()]
    initial_energies = [structure.info['total_energy'] for structure in initial_structures]

    # verify by inspection that total energy has decreased in all instances
    for initial_energy, relaxed_energy in zip(initial_energies, relaxed_energies):
        print(f"Initial energy: {initial_energy} eV, relaxed energy: {relaxed_energy} eV")

    ########################## relaxation one by one ##########################
    from orb_models.forcefield.calculator import ORBCalculator
    calc = ORBCalculator(potential, device=device)

    s_time = time()
    for atoms in atoms_list:
        atoms.calc = calc
        optimizer = FrechetCellFilter(atoms)
        FIRE(optimizer).run(fmax=0.01, steps=1000)  # cause the BatchRelaxer do not limit steps
    e_time = time()
    print(f"Time: {e_time-s_time}")
