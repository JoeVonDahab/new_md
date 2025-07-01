from easy_md.utils.config import create_config
from easy_md.main import run_solvation, run_forcefield_parameterization, run_energy_minimization, run_simulation

config = create_config(
    protein_file="protien/2GQG_one_chain_docking.pdb",
    ligand_file="molecules/028_Crizotinib.sdf",

    # MD simulation settings. See "Simulation Parameters" section below for all options
    md_steps=1000,
    md_save_interval=10,
    
    # Platform settings
    platform_name="CUDA",         # Use "CUDA" for GPU, "CPU" for CPU
    platform_precision="mixed",   # or "single" or "double"
)
run_solvation.add_water(config=config)
run_forcefield_parameterization.main(config)
run_energy_minimization.main(config)
run_simulation.main(config)
# By default `run_simulation.main(config)` loads the energy-minimized state
# saved in `emin.xml`.  To resume a previous run instead, supply the path
# to its state file .xml starting_state_path="path/to/state.xml" or checkpoint file: starting_state_path="path/to/state.xml":