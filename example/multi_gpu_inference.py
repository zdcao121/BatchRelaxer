import math
import torch
from time import time
from ase.build import bulk
from joblib import Parallel, delayed

from orb_models.forcefield import atomic_system
from orb_models.forcefield.base import batch_graphs


# make pytorch deterministic
torch.manual_seed(0)


def make_orb_calc(model_name, model_path, device="cuda"):
    from orb_models.forcefield import pretrained

    if model_name in pretrained.ORB_PRETRAINED_MODELS:
        # Load the ORB forcefield model
        orbff = pretrained.ORB_PRETRAINED_MODELS[model_name](model_path, device=device)

    else:
        raise ValueError(f"Model {model_name} not recognized. Available models: {list(pretrained.ORB_PRETRAINED_MODELS.keys())}")

    return orbff



def infer_on_device(chunk, device):
    graph_batch = batch_graphs(
        [atomic_system.ase_atoms_to_atom_graphs(atoms, potential.system_config, device=device)
            for atoms in chunk]
    )
    res = potential.to(device).predict(graph_batch)
    energy_batch = res['energy']
    forces_batch = res.get("forces", res.get("grad_forces"))
    stress_batch = res.get("stress", res.get("grad_stress"))
    return energy_batch, forces_batch, stress_batch


if __name__ == "__main__":
    ######################## SETUP MODEL ########################
    print("Available GPUs:", torch.cuda.device_count())
    potential = make_orb_calc("orb-v3-conservative-inf-mpa",
                            "./data/ckpt/orb-v3-conservative-inf-mpa-20250404.ckpt",
                                device="cuda")


    ######################## REFERENCE SINGLE-GPU INFERENCE ########################
    # Here, we generate a list of ASE Atoms objects we want to relax
    atoms_list = [bulk("C"), bulk("Mg"), bulk("Si"), bulk("Ni")]

    # And then perturb them a bit so that relaxation is not trivial
    for atoms in atoms_list:
        atoms.rattle(stdev=0.1)

    s_time = time()
    graph_batch = batch_graphs([atomic_system.ase_atoms_to_atom_graphs(atoms, potential.system_config, device="cuda:1") for atoms in atoms_list])
    # print(type(graph_batch))    
    ref_result = potential.to("cuda:1").predict(graph_batch)
    ref_energy_batch = ref_result['energy']
    ref_forces_batch = ref_result.get("forces", ref_result.get("grad_forces"))
    ref_stress_batch = ref_result.get("stress", ref_result.get("grad_stress"))

    # transfer to cpu for later comparison
    ref_energy_batch = ref_energy_batch.cpu()
    ref_forces_batch = ref_forces_batch.cpu()
    ref_stress_batch = ref_stress_batch.cpu()
    e_time = time()
    print(f"Single-GPU inference time for {len(atoms_list)} systems: {e_time - s_time:.2f} seconds")


    ######################## MULTI-GPU INFERENCE ########################
    # Now we want to do the same inference using multiple GPUs
    s_time = time()
    ngpu = torch.cuda.device_count()
    print(f"Using {ngpu} GPUs for inference.")
    chunk_size = math.ceil(len(atoms_list) / ngpu)
    chunks = [atoms_list[i*chunk_size : (i+1)*chunk_size] for i in range(ngpu)]
    chunk_indices = [list(range(i*chunk_size, min((i+1)*chunk_size, len(atoms_list)))) for i in range(ngpu)]

    results = Parallel(n_jobs=ngpu)(
        delayed(infer_on_device)(chunks[i], f"cuda:{i}") for i in range(ngpu)
    )

    # Unpack results
    energies, forces, stresses = zip(*results)

    # cpu transfer
    energies = [e.cpu() for e in energies]
    forces = [f.cpu() for f in forces]
    stresses = [s.cpu() for s in stresses]

    # print(forces[0].shape, forces[1].shape)

    # Combine results from all GPUs
    energy_batch = torch.cat(energies, dim=0)
    forces_batch = torch.cat(forces, dim=0)
    stress_batch = torch.cat(stresses, dim=0)
    e_time = time()
    print(f"Multi-GPU inference time for {len(atoms_list)} systems: {e_time - s_time:.2f} seconds")

    ######################## VERIFY RESULTS ########################
    # Verify that the multi-GPU results match the reference single-GPU results
    assert torch.allclose(ref_energy_batch, energy_batch, atol=1e-3), "Multi-GPU energy results do not match single-GPU results!"
    assert torch.allclose(ref_forces_batch, forces_batch, atol=1e-3), "Multi-GPU forces results do not match single-GPU results!"
    assert torch.allclose(ref_stress_batch, stress_batch, atol=1e-3), "Multi-GPU stress results do not match single-GPU results!"
    print("Multi-GPU inference successful and results match single-GPU inference.")
