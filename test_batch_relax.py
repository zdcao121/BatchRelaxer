import numpy as np
from ase.calculators.calculator import Calculator
from ase.build import bulk

from orb_models.forcefield import atomic_system, pretrained
from orb_models.forcefield.base import batch_graphs
from orb_models.forcefield.calculator import ORBCalculator


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


if __name__ == "__main__":
    import torch
    # Load the ORB-v2 model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    orbff = pretrained.orb_v2("./data/orb-v2-20241011.ckpt", device=device)
    calc = ORBCalculator(orbff, device=device)

    # Here, we generate a list of ASE Atoms objects we want to relax
    atoms_list = [bulk("C"), bulk("Mg"), bulk("Si"), bulk("Ni")]

    # And then perturb them a bit so that relaxation is not trivial
    for atoms in atoms_list:
        atoms.rattle(stdev=0.1)

    ref_energies = []
    ref_forces = []
    ref_stresses = []
    for atoms in atoms_list:
        atoms.calc = calc
        ref_energies.append(atoms.get_potential_energy())
        ref_forces.append(atoms.get_forces())
        ref_stresses.append(atoms.get_stress())

    graph_batch = batch_graphs([atomic_system.ase_atoms_to_atom_graphs(atoms, orbff.system_config, device=device) for atoms in atoms_list])    
    result = orbff.predict(graph_batch)

    energy_batch, forces_batch, stress_batch = result['energy'], result['forces'], result['stress']
    energy_batch = energy_batch.detach().cpu().numpy()
    forces_batch = forces_batch.detach().cpu().numpy()
    stress_batch = stress_batch.detach().cpu().numpy()

    atom_lengths = np.array([len(atoms) for atoms in atoms_list])
    cumulative_lengths = np.cumsum(atom_lengths)  # Cumulative sum of the lengths
    # split the forces into the individual atoms
    forces_batch = np.split(forces_batch, cumulative_lengths[:-1])

    # pointer = 0
    for idx, atoms in enumerate(atoms_list):
        atoms.calc = DummyBatchCalculator()
        # Set the properties so the dummy calculator can
        # return them within the optimizer step
        atoms.info["total_energy"] = energy_batch[idx]
        atoms.info["stress"] = stress_batch[idx]
        atoms.arrays["forces"] = forces_batch[idx]

        assert np.allclose(ref_energies[idx], atoms.get_potential_energy(), atol=1e-2)
        assert np.allclose(ref_forces[idx], atoms.get_forces(), atol=1e-2)
        assert np.allclose(ref_stresses[idx], atoms.get_stress(), atol=1e-2)
        print(f"Test passed for atoms {idx}")
