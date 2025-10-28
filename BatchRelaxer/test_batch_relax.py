import torch
import numpy as np
from ase.calculators.calculator import Calculator
from ase.build import bulk

from orb_models.forcefield import atomic_system, pretrained
from orb_models.forcefield.base import batch_graphs
from orb_models.forcefield.calculator import ORBCalculator

from BatchRelaxer.batch_relax import DummyBatchCalculator


def test_orb(model_ckpt=None):
    """
    Test consistency between per-atom ORB calculator outputs and
    batched ORB model predictions.

    Args:
        model_ckpt (str): Path to the pretrained ORB checkpoint.
    """
    # Load the ORB-v2 model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_ckpt is not None:
        orbff = pretrained.orb_v3_conservative_inf_mpa(model_ckpt, device=device)
    else:
        orbff = pretrained.orb_v3_conservative_inf_mpa(device=device)
    # orbff = pretrained.orb_v3_conservative_inf_mpa("./data/orb-v3-conservative-inf-mpa-20250404.ckpt", device=device)
    # orbff = pretrained.orb_v3_direct_20_omat("data/orb-v3-direct-20-omat-20250404.ckpt", device=device)
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

    energy_batch = result['energy']
    forces_batch = result.get("forces", result.get("grad_forces"))
    stress_batch = result.get("stress", result.get("grad_stress"))
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


if __name__ == "__main__":
    test_orb(model_ckpt="./data/orb-v3-conservative-inf-mpa-20250404.ckpt")
