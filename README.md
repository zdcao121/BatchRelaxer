## Batch relax using Orb models

Orb-models: https://github.com/orbital-materials/orb-models

## Installation
```bash
pip install .
```

## Usage

NOTE: `ase` <= 3.25.0 !

```python
from time import time
from ase.build import bulk
from orb_models.forcefield import pretrained

from BatchRelaxer import BatchRelaxer


device = "cuda"  # or device="cuda"
potential = pretrained.orb_v2("./data/orb-v2-20241011.ckpt", device=device)

# initialize the batch relaxer with a EXPCELLFILTER for cell relaxation and a FIRE optimizer
relaxer = BatchRelaxer(potential,
                       device=device,
                       fmax=0.01,
                       max_n_steps=200,
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

```

`BatchRelaxer` Parameters:
- `potential`: orb models
- `device` : `cpu` or `cuda`
- `optimizer` : Optimization algorithm, e.g., `FIRE` or `BFGS`
- `filter` : Cell relaxation filter, e.g., `FRECHETCELLFILTER` or `EXPCELLFILTER`
- `fmax`: maximum force convergence criterion
- `max_natoms_per_batch` : maximum number of atoms per batch
- `max_n_steps`: maximum number of optimization steps

## Custom BatchRelaxer
If you want to write your own BatchRelaxer for other force fields, you can refer to the following code snippet from the
[file](./BatchRelaxer/batch_relax.py).


```python
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
```

## Reference:
- `batch_relax.py` in [MatterSim](https://github.com/microsoft/mattersim/blob/5b1ee33b615faae41abd56581336827b2a1c49d3/src/mattersim/applications/batch_relax.py)
- torch-sim package