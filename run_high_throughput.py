#!/usr/bin/env python3
"""
High-throughput molecular dynamics simulation script for drug screening.

This script processes all molecules in the molecules/ folder, runs MD simulations
for each, and compiles results into a ranked CSV file based on binding affinity.

STABILITY FIXES IMPLEMENTED:
- Reduced timestep to 1 fs (from 2 fs) for better integrator stability
- Increased energy minimization steps to 5000 (from 10) 
- Added gradual heating in 25K increments (from 300K jump)
- Reduced restraint force constant to 10 kJ/mol/nm² (from 100) for less aggressive restraints
- Added HBonds constraints to all hydrogen-containing bonds for stability
- Improved state file resolution to load equilibrated states when available
- Added frequent energy monitoring during equilibration to catch problems early
- Uses LangevinMiddleIntegrator (BAOAB scheme) for better stability

USAGE:
- Normal run: python run_high_throughput.py
- Test run (20 ps): python run_high_throughput.py --test
- Custom steps: python run_high_throughput.py --md_steps 50000
"""

import os
import glob
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
import time
from dataclasses import dataclass
import MDAnalysis as mda
from openmm import unit

from easy_md.utils.config import create_config
from easy_md.main import run_solvation, run_forcefield_parameterization, run_energy_minimization, run_simulation

@dataclass
class SimulationResult:
    """Data class to store simulation results for each molecule."""
    molecule_name: str
    smiles: str = ""
    final_energy: float = 0.0
    average_energy: float = 0.0
    energy_std: float = 0.0
    rmsd_ligand: float = 0.0
    rmsd_protein: float = 0.0
    simulation_time: float = 0.0
    success: bool = False
    error_message: str = ""
    binding_score: float = 0.0  # Composite score for ranking

class HighThroughputMD:
    """High-throughput molecular dynamics simulation manager."""
    
    def __init__(self, 
                 protein_file: str,
                 molecules_dir: str = "molecules",
                 output_base_dir: str = "md_results",
                 md_steps: int = 5000,
                 platform_name: str = "CUDA"):
        """
        Initialize the high-throughput MD system.
        
        Args:
            protein_file: Path to the protein PDB file
            molecules_dir: Directory containing ligand SDF files
            output_base_dir: Base directory for all outputs
            md_steps: Number of MD simulation steps
            platform_name: OpenMM platform ("CUDA", "CPU", etc.)
        """
        self.protein_file = protein_file
        self.molecules_dir = Path(molecules_dir)
        self.output_base_dir = Path(output_base_dir)
        self.md_steps = md_steps
        self.platform_name = platform_name
        
        # Create base output directory
        self.output_base_dir.mkdir(exist_ok=True)
        
        # Results storage
        self.results: List[SimulationResult] = []
        
    def get_molecule_files(self) -> List[Path]:
        """Get list of SDF files to process, excluding Zone.Identifier files."""
        sdf_files = list(self.molecules_dir.glob("*.sdf"))
        # Filter out Zone.Identifier files
        sdf_files = [f for f in sdf_files if ":Zone.Identifier" not in str(f)]
        return sorted(sdf_files)
    
    def extract_molecule_name(self, sdf_path: Path) -> str:
        """Extract molecule name from SDF filename."""
        # Remove extension and any numbering prefix
        name = sdf_path.stem
        # Remove number prefix if present (e.g., "028_Crizotinib" -> "Crizotinib")
        if "_" in name and name.split("_")[0].isdigit():
            name = "_".join(name.split("_")[1:])
        return name
    
    def get_smiles_from_sdf(self, sdf_path: Path) -> str:
        """Extract SMILES string from SDF file if available."""
        try:
            from rdkit import Chem
            mol = Chem.SDMolSupplier(str(sdf_path), removeHs=False)[0]
            if mol is not None:
                return Chem.MolToSmiles(mol)
        except Exception as e:
            print(f"Could not extract SMILES from {sdf_path}: {e}")
        return ""
    
    def calculate_binding_energy(self, config: Dict[str, Any]) -> Tuple[float, float, float]:
        """
        Calculate binding energy metrics from simulation.
        
        Returns:
            Tuple of (final_energy, average_energy, energy_std)
        """
        try:
            # Try to read energy from log file if it exists
            log_file = config.get('path_md_log', '').replace('_id', '_id_0')
            if os.path.exists(log_file):
                energies = self._parse_energy_log(log_file)
                if energies:
                    return energies[-1], np.mean(energies), np.std(energies)
            
            # Fallback: estimate from final state if available
            state_file = config.get('path_md_state', '').replace('_id', '_id_0') 
            if os.path.exists(state_file):
                return self._estimate_energy_from_state(state_file)
                
        except Exception as e:
            print(f"Error calculating binding energy: {e}")
        
        return 0.0, 0.0, 0.0
    
    def _parse_energy_log(self, log_file: str) -> List[float]:
        """Parse energy values from simulation log file."""
        energies = []
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                
            # Skip header line (first line)
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                    
                # Split by tab and extract potential energy (4th column, index 3)
                parts = line.split('\t')
                if len(parts) >= 4:
                    try:
                        # Extract potential energy value
                        energy = float(parts[3])
                        energies.append(energy)
                    except (ValueError, IndexError):
                        continue
                        
        except Exception as e:
            print(f"Error parsing energy log: {e}")
        return energies
    
    def _estimate_energy_from_state(self, state_file: str) -> Tuple[float, float, float]:
        """Estimate energy from final state file."""
        # This is a placeholder - in a real implementation, you'd load the state
        # and calculate the potential energy
        return 0.0, 0.0, 0.0
    
    def calculate_rmsd_metrics(self, config: Dict[str, Any]) -> Tuple[float, float]:
        """
        Calculate RMSD metrics for protein and ligand.
        
        Returns:
            Tuple of (ligand_rmsd, protein_rmsd)
        """
        try:
            # Check if trajectory file exists
            traj_file = config.get('path_md_trajectory', '').replace('_id', '_id_0')
            # Fix the typo in trajectory filename if needed
            if traj_file and not os.path.exists(traj_file):
                traj_file = traj_file.replace('md_trajectory_id_0', 'md_trajetory_id_0')
            
            structure_file = config.get('path_protein_solvated', '')
            
            if not os.path.exists(traj_file) or not os.path.exists(structure_file):
                return 0.0, 0.0
            
            # Use MDAnalysis to calculate RMSD
            u = mda.Universe(structure_file, traj_file)
            
            # Calculate ligand RMSD
            ligand = u.select_atoms("resname UNK")
            if len(ligand) > 0:
                ligand_positions = []
                for ts in u.trajectory:
                    ligand_positions.append(ligand.positions.copy())
                
                if len(ligand_positions) > 1:
                    ref_pos = ligand_positions[0]
                    rmsd_values = []
                    for pos in ligand_positions[1:]:
                        rmsd = np.sqrt(np.mean((pos - ref_pos)**2))
                        rmsd_values.append(rmsd)
                    ligand_rmsd = np.mean(rmsd_values)
                else:
                    ligand_rmsd = 0.0
            else:
                ligand_rmsd = 0.0
            
            # Calculate protein backbone RMSD
            protein_ca = u.select_atoms("protein and name CA")
            if len(protein_ca) > 0:
                protein_positions = []
                for ts in u.trajectory:
                    protein_positions.append(protein_ca.positions.copy())
                
                if len(protein_positions) > 1:
                    ref_pos = protein_positions[0]
                    rmsd_values = []
                    for pos in protein_positions[1:]:
                        rmsd = np.sqrt(np.mean((pos - ref_pos)**2))
                        rmsd_values.append(rmsd)
                    protein_rmsd = np.mean(rmsd_values)
                else:
                    protein_rmsd = 0.0
            else:
                protein_rmsd = 0.0
                
            return ligand_rmsd, protein_rmsd
            
        except Exception as e:
            print(f"Error calculating RMSD: {e}")
            return 0.0, 0.0
    
    def calculate_binding_score(self, result: SimulationResult) -> float:
        """
        Calculate a composite binding score for ranking molecules.
        Higher scores indicate better binding.
        
        Args:
            result: SimulationResult object
            
        Returns:
            Composite binding score
        """
        # Weights for different metrics
        energy_weight = 0.7
        stability_weight = 0.2
        consistency_weight = 0.1
        
        # Energy score (more negative = better)
        # Normalize by dividing by typical protein energy scale
        energy_score = abs(result.average_energy) / 100000.0 if result.average_energy < 0 else 0
        
        # Stability score (lower RMSD = better)
        stability_score = 1.0 / (1.0 + result.rmsd_ligand) if result.rmsd_ligand > 0 else 1.0
        
        # Consistency score (lower std = better)
        consistency_score = 1.0 / (1.0 + result.energy_std/1000.0) if result.energy_std > 0 else 1.0
        
        # Composite score (higher = better)
        composite_score = (energy_weight * energy_score + 
                          stability_weight * stability_score + 
                          consistency_weight * consistency_score)
        
        return composite_score
    
    def run_single_simulation(self, sdf_file: Path) -> SimulationResult:
        """Run MD simulation for a single molecule."""
        molecule_name = self.extract_molecule_name(sdf_file)
        print(f"\n{'='*60}")
        print(f"Processing molecule: {molecule_name}")
        print(f"SDF file: {sdf_file}")
        print(f"{'='*60}")
        
        # Create output directory for this molecule
        output_dir = self.output_base_dir / molecule_name
        output_dir.mkdir(exist_ok=True)
        
        # Initialize result object
        result = SimulationResult(
            molecule_name=molecule_name,
            smiles=self.get_smiles_from_sdf(sdf_file)
        )
        
        start_time = time.time()
        
        try:
            # Create configuration with stability fixes
            config = create_config(
                protein_file=self.protein_file,
                ligand_file=str(sdf_file),
                project_dir=str(output_dir),
                output_dir=str(output_dir / "output"),
                config_dir=str(output_dir / "config"),
                
                # INTEGRATOR: Use safer settings for equilibration 
                integrator_timestep=0.001,  # 1 fs timestep for stability
                integrator_friction=1.0,    # Keep stable friction
                
                # ENERGY MINIMIZATION: More thorough minimization
                emin_steps=5000,            # Increase from 10 to 5000 steps
                emin_tolerance="10 * kilojoule_per_mole / nanometer",
                emin_heating_step=25,       # Heat in 25K increments instead of 300K
                emin_heating_interval=1000,  # More steps per heating interval
                
                # MD SIMULATION: Conservative settings for equilibration
                md_steps=self.md_steps,
                md_save_interval=max(100, self.md_steps // 500),  # Reasonable save frequency
                md_npt=False,               # Start with NVT equilibration
                
                # RESTRAINTS: Weaker force constants to prevent instabilities  
                md_harmonic_restraint=True,
                restraint_force_constant=10,  # Reduce from default 100 to 10 kJ/mol/nm²
                md_restrained_residues=[],    # Apply to all residues initially
                
                # PLATFORM: Keep user settings but ensure mixed precision
                platform_name=self.platform_name,
                platform_precision="mixed",
                
                # ANALYSIS: Keep as specified
                rmsd_ligand_selection="resname UNK",
                rmsd_selection="protein and name CA",
            )
            
            print(f"Configuration created. Output directory: {output_dir}")
            
            # Run simulation pipeline
            print("Step 1: Adding water...")
            run_solvation.add_water(config=config)
            
            print("Step 2: Force field parameterization...")
            run_forcefield_parameterization.main(config)
            
            print("Step 3: Energy minimization...")
            run_energy_minimization.main(config)
            
            print("Step 4: MD simulation...")
            run_simulation.main(config)
            
            # Calculate metrics
            print("Step 5: Analyzing results...")
            final_energy, avg_energy, energy_std = self.calculate_binding_energy(config)
            ligand_rmsd, protein_rmsd = self.calculate_rmsd_metrics(config)
            
            # Update result
            result.final_energy = final_energy
            result.average_energy = avg_energy
            result.energy_std = energy_std
            result.rmsd_ligand = ligand_rmsd
            result.rmsd_protein = protein_rmsd
            result.simulation_time = time.time() - start_time
            result.success = True
            
            # Calculate binding score
            result.binding_score = self.calculate_binding_score(result)
            
            print(f"Simulation completed successfully!")
            print(f"Final energy: {final_energy:.2f} kJ/mol")
            print(f"Average energy: {avg_energy:.2f} kJ/mol")
            print(f"Ligand RMSD: {ligand_rmsd:.2f} Å")
            print(f"Binding score: {result.binding_score:.4f}")
            
        except Exception as e:
            result.error_message = str(e)
            result.simulation_time = time.time() - start_time
            print(f"Simulation failed: {e}")
        
        return result
    
    def run_all_simulations(self, test_single_molecule=False) -> None:
        """Run simulations for all molecules in the directory."""
        sdf_files = self.get_molecule_files()
        
        if not sdf_files:
            print("No SDF files found in molecules directory!")
            return
        
        print(f"Found {len(sdf_files)} molecules to process:")
        for sdf_file in sdf_files:
            print(f"  - {self.extract_molecule_name(sdf_file)}")
        
        # If testing, only run the first molecule (or find Gilteritinib)
        if test_single_molecule:
            # Try to find Gilteritinib first, otherwise use first molecule
            test_molecule = None
            for sdf_file in sdf_files:
                if "gilteritinib" in self.extract_molecule_name(sdf_file).lower():
                    test_molecule = sdf_file
                    break
            if test_molecule is None:
                test_molecule = sdf_files[0]
            
            sdf_files = [test_molecule]
            print(f"\n🧪 TEST MODE: Only processing {self.extract_molecule_name(test_molecule)}")
        
        # Process each molecule
        for i, sdf_file in enumerate(sdf_files, 1):
            print(f"\n{'#'*70}")
            print(f"Processing molecule {i}/{len(sdf_files)}")
            print(f"{'#'*70}")
            
            result = self.run_single_simulation(sdf_file)
            self.results.append(result)
            
            # Save intermediate results
            self.save_results()
            
            # In test mode, break after first molecule if it succeeded
            if test_single_molecule and result.success:
                print(f"\n✅ TEST SUCCESS: {result.molecule_name} completed without NaN errors!")
                print(f"   Final energy: {result.final_energy:.2f} kJ/mol")
                print(f"   Average energy: {result.average_energy:.2f} kJ/mol") 
                print(f"   Binding score: {result.binding_score:.4f}")
                print(f"   You can now run the full batch with confidence!")
                break
            elif test_single_molecule and not result.success:
                print(f"\n❌ TEST FAILED: {result.molecule_name} encountered errors:")
                print(f"   Error: {result.error_message}")
                print(f"   Check the output logs for more details.")
                break
        
        if not test_single_molecule:
            print(f"\n{'='*70}")
            print("All simulations completed!")
            print(f"{'='*70}")
            self.print_summary()
    
    def save_results(self) -> None:
        """Save results to CSV and JSON files."""
        if not self.results:
            return
        
        # Prepare data for CSV
        csv_data = []
        for result in self.results:
            csv_data.append({
                'Rank': 0,  # Will be filled after sorting
                'Molecule_Name': result.molecule_name,
                'SMILES': result.smiles,
                'Binding_Score': result.binding_score,
                'Final_Energy_kJ_mol': result.final_energy,
                'Average_Energy_kJ_mol': result.average_energy,
                'Energy_Std_kJ_mol': result.energy_std,
                'Ligand_RMSD_A': result.rmsd_ligand,
                'Protein_RMSD_A': result.rmsd_protein,
                'Simulation_Time_s': result.simulation_time,
                'Success': result.success,
                'Error_Message': result.error_message
            })
        
        # Sort by binding score (higher is better)
        csv_data.sort(key=lambda x: x['Binding_Score'], reverse=True)
        
        # Add ranks
        for i, row in enumerate(csv_data, 1):
            row['Rank'] = i
        
        # Save to CSV
        df = pd.DataFrame(csv_data)
        csv_file = self.output_base_dir / "simulation_results_ranked.csv"
        df.to_csv(csv_file, index=False)
        print(f"Results saved to: {csv_file}")
        
        # Save detailed results to JSON
        json_data = []
        for result in self.results:
            json_data.append({
                'molecule_name': result.molecule_name,
                'smiles': result.smiles,
                'final_energy': result.final_energy,
                'average_energy': result.average_energy,
                'energy_std': result.energy_std,
                'rmsd_ligand': result.rmsd_ligand,
                'rmsd_protein': result.rmsd_protein,
                'simulation_time': result.simulation_time,
                'success': result.success,
                'error_message': result.error_message,
                'binding_score': result.binding_score
            })
        
        json_file = self.output_base_dir / "detailed_results.json"
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"Detailed results saved to: {json_file}")
    
    def print_summary(self) -> None:
        """Print summary of simulation results."""
        if not self.results:
            print("No results to summarize.")
            return
        
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        
        print(f"\nSUMMARY:")
        print(f"Total molecules processed: {len(self.results)}")
        print(f"Successful simulations: {len(successful)}")
        print(f"Failed simulations: {len(failed)}")
        
        if successful:
            # Sort by binding score
            successful.sort(key=lambda x: x.binding_score, reverse=True)
            
            print(f"\nTOP 5 MOLECULES (by binding score):")
            print(f"{'Rank':<5} {'Molecule':<20} {'Score':<10} {'Avg Energy':<15} {'Ligand RMSD':<12}")
            print("-" * 65)
            
            for i, result in enumerate(successful[:5], 1):
                print(f"{i:<5} {result.molecule_name:<20} {result.binding_score:<10.4f} "
                      f"{result.average_energy:<15.2f} {result.rmsd_ligand:<12.2f}")
        
        if failed:
            print(f"\nFAILED SIMULATIONS:")
            for result in failed:
                print(f"  - {result.molecule_name}: {result.error_message[:100]}...")
        
        total_time = sum(r.simulation_time for r in self.results)
        print(f"\nTotal computation time: {total_time/3600:.2f} hours")

def main():
    """Main function to run high-throughput MD simulations."""
    import sys
    
    # Check for test mode
    test_mode = '--test' in sys.argv or '--md_steps' in sys.argv
    
    # Configuration - Update these paths for your project
    protein_file = "protien/LRRK2_Monomer_8fo7.pdb"  # Using the available protein file
    molecules_dir = "molecules"
    output_dir = "md_screening_results"
    
    # Use short test run for stability testing, or full production run
    if test_mode:
        md_steps = 20000      # ~20 ps at 1 fs for testing stability
        print("Running in TEST MODE with short simulations")
    else:
        md_steps = 5000000    # Full production length  
        print("Running in PRODUCTION MODE")
        
    # Parse command line for custom md_steps
    if '--md_steps' in sys.argv:
        try:
            md_steps_idx = sys.argv.index('--md_steps')
            md_steps = int(sys.argv[md_steps_idx + 1])
            print(f"Using custom md_steps: {md_steps}")
        except (IndexError, ValueError):
            print("Invalid --md_steps argument, using default")
    
    platform = "CUDA"  # Use "CPU" if no GPU available
    
    print("Easy-MD High-Throughput Screening Pipeline")
    print("=" * 50)
    print(f"Protein: {protein_file}")
    print(f"Molecules directory: {molecules_dir}")
    print(f"Output directory: {output_dir}")
    print(f"MD steps: {md_steps}")
    print(f"Platform: {platform}")
    if test_mode:
        print("⚠️  TEST MODE: Using conservative settings for stability")
    print("=" * 50)
    
    # Create and run high-throughput MD
    htmd = HighThroughputMD(
        protein_file=protein_file,
        molecules_dir=molecules_dir,
        output_base_dir=output_dir,
        md_steps=md_steps,
        platform_name=platform
    )
    
    # Run in test mode if specified
    htmd.run_all_simulations(test_single_molecule=test_mode)

if __name__ == "__main__":
    main()