# In your package (e.g., src/easy_md/config.py)
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Default configuration values - using prefixes for grouping
DEFAULT_CONFIG = {
    # Paths
    "path_base": "",  # Will be set in create_config
    "path_protein": "",  # Will be set in create_config
    "path_ligand": "",  # Will be set in create_config
    "path_protein_solvated": "",  # Will be set in create_config
    "path_openff_topology": "",  # Will be set in create_config
    "path_openff_interchange": "",  # Will be set in create_config
    "path_openmm_topology": "",  # Will be set in create_config
    "path_openmm_system": "",  # Will be set in create_config
    "path_emin_structure": "",  # Will be set in create_config
    "path_emin_state": "",  # Will be set in create_config
    "path_md_log": "",  # Will be set in create_config
    "path_md_trajectory": "",  # Will be set in create_config
    "path_md_checkpoint": "",  # Will be set in create_config
    "path_md_state": "",  # Will be set in create_config
    "path_rmsd_output": "",  # Will be set in create_config
    "path_rmsd_ligand_output": "",  # Will be set in create_config
    "path_rmsf_output": "",  # Will be set in create_config
    "path_amber_topology": "",  # Will be set in create_config
    
    # Forcefields
    "ff_small_molecule_openff": "openff-2.0.0.offxml",
    "ff_protein_openff": "ff14sb_off_impropers_0.0.3.offxml",
    "ff_protein": "amber14-all.xml",
    "ff_water": "amber14/tip3pfb.xml",
    
    # Integrator
    "integrator_temperature": 300.0,
    "integrator_friction": 1.0,
    "integrator_timestep": 0.002,
    "integrator_type": "langevin_middle",  # Options: "langevin_middle", "langevin"
    
    # Constraints
    "add_constraints": True,       # Add HBonds constraints for stability  
    "rigid_water": True,          # Use rigid water molecules

    # Solvation
    "solv_box_buffer": 2.5, # angstroms
    "solv_ionic_strength": 0.15, # molar
    "solv_positive_ion": "Na+", # the type of positive ion to add. Allowed values are 'Cs+', 'K+', 'Li+', 'Na+', and 'Rb+'
    "solv_negative_ion": "Cl-", # the type of negative ion to add. Allowed values are 'Cl-', 'Br-', 'F-', and 'I-'. Be aware that not all force fields support all ion types.
    "solv_model": "tip3p", # Supported values are 'tip3p', 'spce', 'tip4pew', and 'tip5p'.
    "solv_pH": 7.0, # pH of the solvent
    
    # MD Simulation
    "md_steps": 1000,
    "md_save_interval": 10,
    "md_pressure": 1.0,
    "md_anisotropic": False,
    "md_barostat_freq": 25,
    "md_harmonic_restraint": True,
    "md_load_state": True,
    "md_restrained_residues": [],
    "md_npt": False,
    
    # Restraints
    "restraint_force_constant": 100,  # kJ/mol/nm^2 - reduce to 10 for stability
    
    # Monitor
    "monitor_window": 10,
    "monitor_temp_threshold": 2.0,
    "monitor_energy_threshold": 100.0,
        
    # Platform
    "platform_name": "CPU",
    "platform_precision": "mixed",
    
    # Analysis - RMSD
    "rmsd_selection": "protein and name CA",
    "rmsd_ligand_selection": "resname UNK",
    
    # Analysis - RMSF
    "rmsf_selection": "protein and name CA",
    
    # Energy minimization
    "emin_tolerance": "5 * kilojoule_per_mole / nanometer",
    "emin_heating_step": 300,
    "emin_target_temp": 300,
    "emin_heating_interval": 1,
    "emin_steps": 10
}

def create_config(
    protein_file: str = None,
    ligand_file: str = None,
    project_dir: str = None,
    output_dir: str = None,
    config_dir: str = None,
    save_config_as: str = "simulation_config.yaml",
    **params
) -> Dict[str, Any]:
    """Create a simulation configuration with user-provided settings.

    Args:
        protein_file (str): Path to protein PDB file
        ligand_file (str, optional): Path to ligand file
        project_dir (str, optional): Main project directory. If None, uses protein file directory
        output_dir (str, optional): Output directory. If None, creates 'output' in project_dir
        config_dir (str, optional): Directory to save config file. If None, saves in project_dir
        save_config_as (str, optional): Name of config file. Defaults to "simulation_config.yaml"
        **params: Override any default settings using the following parameters:

            Forcefields:
                ff_small_molecule_openff (str): Forcefield for small molecules. Default: "openff-2.0.0.offxml"
                ff_protein_openff (str): Forcefield for protein. Default: "ff14sb_off_impropers_0.0.3.offxml"
                ff_protein (str): Forcefield for protein. Default: "amber14-all.xml"
                ff_water (str): Forcefield for water. Default: "amber14/tip3pfb.xml"

            Integrator Settings:
                integrator_temperature (float): Temperature in Kelvin. Default: 300.0
                integrator_friction (float): Friction coefficient in ps^-1. Default: 1.0
                integrator_timestep (float): Time step in ps. Default: 0.002
                integrator_type (str): Integrator type. Default: "langevin_middle"
                                     Options: ["langevin_middle", "langevin"]
                                     
            Constraints Settings:
                add_constraints (bool): Add HBonds constraints. Default: True
                rigid_water (bool): Use rigid water molecules. Default: True

            Solvation Settings:
                solv_box_buffer (float): Buffer size in angstroms. Default: 2.5
                solv_ionic_strength (float): Ionic strength in molar. Default: 0.15
                solv_positive_ion (str): Type of positive ion. Default: "Na+"
                                       Allowed: ['Cs+', 'K+', 'Li+', 'Na+', 'Rb+']
                solv_negative_ion (str): Type of negative ion. Default: "Cl-"
                                       Allowed: ['Cl-', 'Br-', 'F-', 'I-']
                solv_model (str): Water model. Default: "tip3p"
                                Allowed: ['tip3p', 'spce', 'tip4pew', 'tip5p']
                solv_pH (float): pH of the solvent. Default: 7.0

            MD Simulation Settings:
                md_steps (int): Total simulation steps. Default: 1000
                md_save_interval (int): Save interval for trajectory. Default: 10
                md_pressure (float): Pressure in atmospheres. Default: 1.0
                md_anisotropic (bool): Use anisotropic pressure. Default: False
                md_barostat_freq (int): Barostat frequency. Default: 25
                md_harmonic_restraint (bool): Use harmonic restraints. Default: True
                md_load_state (bool): Load previous state if available. Default: True
                md_restrained_residues (list): List of residues to restrain. Default: []
                md_npt (bool): Use NPT ensemble. Default: False
                
            Restraint Settings:
                restraint_force_constant (float): Force constant for harmonic restraints in kJ/mol/nm^2. 
                                                 Default: 100. Reduce to 10 for better stability.

            Monitoring Settings:
                monitor_window (int): Window size for monitoring. Default: 10
                monitor_temp_threshold (float): Temperature threshold. Default: 2.0
                monitor_energy_threshold (float): Energy threshold. Default: 100.0

            Platform Settings:
                platform_name (str): Platform for computation. Default: "CPU"
                                   Allowed: ['CPU', 'CUDA']
                platform_precision (str): Precision mode. Default: "mixed"
                                        Allowed: ['single', 'mixed', 'double']

            Analysis Settings:
                rmsd_selection (str): Atom selection for RMSD. Default: "protein and name CA"
                rmsd_ligand_selection (str): Atom selection for ligand RMSD. Default: "resname UNK"
                rmsf_selection (str): Atom selection for RMSF. Default: "protein and name CA"

            Energy Minimization Settings:
                emin_tolerance (str): Energy minimization tolerance. 
                                    Default: "5 * kilojoule_per_mole / nanometer"
                emin_heating_step (int): Heating step size. Default: 300
                emin_target_temp (float): Target temperature. Default: 300
                emin_heating_interval (int): Steps per heating interval. Default: 1
                emin_steps (int): Total minimization steps. Default: 10

    Returns:
        Dict[str, Any]: Complete simulation configuration

    Example:
        >>> config = create_config(
        ...     protein_file="protein.pdb",
        ...     ligand_file="ligand.mol2",
        ...     platform_name="CUDA",
        ...     integrator_temperature=310.0
        ... )
    """
    # Convert paths to absolute paths

    protein_file = Path(protein_file).absolute()
    project_dir = Path(project_dir).absolute() if project_dir else Path(protein_file).parent.absolute()
    
    if ligand_file:
        ligand_file = Path(ligand_file).absolute()
    
    # Set up directories
    output_dir = Path(output_dir).absolute() if output_dir else project_dir / "output"
    config_dir = Path(config_dir).absolute() if config_dir else project_dir / "config"
    
    # Validate paths
    if not protein_file.exists():
        print(f"Protein file not found: {protein_file}. Upload protein file to folder")
    if ligand_file and not ligand_file.exists():
        print(f"Ligand file not found: {ligand_file}. Upload ligand file to folder")

    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)

    # Start with default configuration
    config = DEFAULT_CONFIG.copy()
    
    # Update with user parameters
    for param, value in params.items():
        if param in config:
            config[param] = value
        else:
            raise ValueError(f"Unknown parameter: {param}")

    # Set up paths
    config.update({
        "path_base": str(project_dir),
        "path_protein": str(protein_file),
        "path_ligand": str(ligand_file) if ligand_file else "",
        "path_protein_solvated": str(output_dir / "protein_solvated.pdb"),
        "path_openff_topology": str(output_dir / "openff_topology.json"),
        "path_openff_interchange": str(output_dir / "openff_interchange.pdb"),
        "path_openmm_topology": str(output_dir / "openmm_topology.pkl"),
        "path_openmm_system": str(output_dir / "openmm_system.xml"),
        "path_emin_structure": str(output_dir / "emin.pdb"),
        "path_emin_state": str(output_dir / "emin.xml"),
        "path_md_log": str(output_dir / "md_id.log"),
        "path_md_trajectory": str(output_dir / "md_trajetory_id.dcd"),
        "path_md_checkpoint": str(output_dir / "md_checkpoint_id.chk"),
        "path_md_state": str(output_dir / "md_state_id.xml"),
        "path_md_image": str(output_dir / "md_image_id.dcd"),
        "path_rmsd_output": str(output_dir / "rmsd.pkl"),
        "path_rmsd_ligand_output": str(output_dir / "rmsd_ligand.pkl"),
        "path_rmsf_output": str(output_dir / "rmsf.log"),
        "path_amber_topology": str(output_dir / "amber_top.prmtop")
    })

    # Save configuration
    config_path = config_dir / save_config_as
    print(f"Saving configuration to: {config_path}")
    
    # Sort keys by prefix for better readability in YAML
    sorted_config = dict(sorted(config.items()))
    
    with open(config_path, 'w') as f:
        yaml.dump(sorted_config, f, default_flow_style=False, sort_keys=False)
    
    # Display the configuration
    print_config(config)
    
    return config

def load_config(config_file: str, print: bool = True) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_file (str): Path to the YAML configuration file
        print (bool): Whether to print the configuration
        
    Returns:
        dict: Configuration dictionary with flat structure
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    if print:
        print_config(config)
        
    return config

def print_config(config: Dict[str, Any]):
    """Print all configuration settings in a readable format.
    
    Args:
        config (dict): Configuration dictionary with flat structure using prefixes
    """
    print("\n=== Simulation Configuration ===\n")
    
    # Group parameters by prefix
    current_prefix = None
    for key in sorted(config.keys()):
        # Get prefix (everything up to first underscore)
        prefix = key.split('_')[0]
        
        # Print section header when prefix changes
        if prefix != current_prefix:
            print(f"\n{prefix.upper()}:")
            current_prefix = prefix
        
        # Get the value and format it
        value = config[key]
        if isinstance(value, (list, tuple)):
            if not value:
                print(f"  {key}: []")
            else:
                print(f"  {key}:")
                for item in value:
                    print(f"    - {item}")
        else:
            print(f"  {key}: {value}")
    
    print()


# Example usage:
project_structure = """
my_project/
├── config/
│   └── simulation_config.yaml
├── structures/
│   ├── protein.pdb
│   └── ligand.sdf
└── output/
"""
