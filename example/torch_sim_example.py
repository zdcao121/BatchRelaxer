import pandas as pd
from pymatgen.core import Structure
from time import time

from ase.build import bulk
from torch_sim.models.orb import OrbModel
from torch_sim.state import initialize_state
import torch_sim as ts
import torch
# ORB model imports
from orb_models.forcefield import pretrained


device = "cuda" if torch.cuda.is_available() else "cpu"
 
# data = pd.read_csv("./data/output_ZnS_struct.csv")
# structures = [Structure.from_dict(eval(cif)) for cif in data['cif']]
# atoms_list = [s.to_ase_atoms() for s in structures]

atoms_list = [bulk("Ti", "hcp", a=2.95, c=4.68)] * 4

init_state = initialize_state(atoms_list, dtype=torch.float32, device=device)

# Load model
orbff = pretrained.orb_v3_conservative_inf_mpa("./data/orb-v3-conservative-inf-mpa-20250404.ckpt", device=device)
orb_model = OrbModel(model=orbff, dtype=torch.get_default_dtype())

batcher = ts.InFlightAutoBatcher(
    model=orb_model,
    memory_scales_with="n_atoms",
    max_memory_scaler=10000,
    max_iterations=200,  # Optional: maximum convergence attempts per state
)

s_time = time()
# relax all of the high temperature states
relaxed_state = ts.optimize(
    system=init_state,
    model=orb_model,
    optimizer=ts.Optimizer.fire,
    autobatcher=batcher,
    init_kwargs=dict(cell_filter=ts.CellFilter.frechet, fmax=0.1, nsteps=200),
)
print(f"Relax time: {time()-s_time}s")
print(relaxed_state.energy)
