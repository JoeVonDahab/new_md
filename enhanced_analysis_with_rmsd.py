#!/usr/bin/env python3
"""
Re-run comprehensive analysis with RMSD using MDAnalysis directly.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import MDAnalysis as mda
from MDAnalysis.analysis import rms
from dataclasses import dataclass

@dataclass
class EnhancedResult:
    molecule_name: str
    final_energy: float = 0.0
    average_energy: float = 0.0
    energy_std: float = 0.0
    ligand_rmsd_avg: float = 0.0
    ligand_rmsd_std: float = 0.0
    protein_rmsd_avg: float = 0.0
    protein_rmsd_std: float = 0.0
    binding_score: float = 0.0
    stability_score: float = 0.0
    overall_score: float = 0.0

def parse_energy_log(log_file):
    """Parse energy from log file."""
    energies = []
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines[1:]:
            if line.strip():
                parts = line.split('\t')
                if len(parts) >= 4:
                    try:
                        energy = float(parts[3])
                        energies.append(energy)
                    except:
                        continue
    except:
        pass
    return energies

def analyze_rmsd(mol_dir):
    """Analyze RMSD for a molecule."""
    structure_file = mol_dir / "output" / "openff_interchange.pdb"
    traj_file = mol_dir / "output" / "md_trajetory_id_0.dcd"
    
    results = {'ligand_rmsd_avg': 0.0, 'ligand_rmsd_std': 0.0, 
               'protein_rmsd_avg': 0.0, 'protein_rmsd_std': 0.0}
    
    if not (structure_file.exists() and traj_file.exists()):
        return results
    
    try:
        u = mda.Universe(str(structure_file), str(traj_file))
        
        # Protein analysis
        protein = u.select_atoms("protein and backbone")
        if len(protein) > 0:
            R_protein = rms.RMSD(protein, ref=protein)
            R_protein.run()
            results['protein_rmsd_avg'] = np.mean(R_protein.results.rmsd[:, 2])
            results['protein_rmsd_std'] = np.std(R_protein.results.rmsd[:, 2])
        
        # Ligand analysis
        ligand = u.select_atoms("resname UNK")
        if len(ligand) == 0:
            ligand = u.select_atoms("not protein and not resname HOH and not resname Na+ and not resname Cl-")
        
        if len(ligand) > 0:
            R_ligand = rms.RMSD(ligand, ref=ligand)
            R_ligand.run()
            results['ligand_rmsd_avg'] = np.mean(R_ligand.results.rmsd[:, 2])
            results['ligand_rmsd_std'] = np.std(R_ligand.results.rmsd[:, 2])
    
    except Exception as e:
        print(f"RMSD analysis error for {mol_dir.name}: {e}")
    
    return results

def calculate_scores(result):
    """Calculate binding and stability scores."""
    # Binding score (energy-based)
    energy_score = abs(result.average_energy) / 100000.0 if result.average_energy < 0 else 0
    
    # Stability scores (RMSD-based)
    ligand_stability = max(0, (3.0 - result.ligand_rmsd_avg) / 3.0) if result.ligand_rmsd_avg > 0 else 0
    protein_stability = max(0, (2.0 - result.protein_rmsd_avg) / 2.0) if result.protein_rmsd_avg > 0 else 0
    
    result.binding_score = energy_score
    result.stability_score = (ligand_stability + protein_stability) / 2.0
    result.overall_score = (0.7 * result.binding_score + 0.3 * result.stability_score)
    
    return result

def main():
    """Run enhanced analysis with RMSD."""
    results_dir = Path("md_screening_results")
    molecules = ["Crizotinib", "Axitinib", "Vandetanib", "Risdiplam"]
    
    results = []
    
    for mol_name in molecules:
        mol_dir = results_dir / mol_name
        print(f"\nAnalyzing {mol_name}...")
        
        result = EnhancedResult(molecule_name=mol_name)
        
        # Parse energy
        log_file = mol_dir / "output" / "md_id_0.log"
        if log_file.exists():
            energies = parse_energy_log(log_file)
            if energies:
                result.final_energy = energies[-1]
                result.average_energy = np.mean(energies)
                result.energy_std = np.std(energies)
                print(f"  Average energy: {result.average_energy:.1f} kJ/mol")
        
        # Analyze RMSD
        print(f"  Running RMSD analysis...")
        rmsd_data = analyze_rmsd(mol_dir)
        result.ligand_rmsd_avg = rmsd_data['ligand_rmsd_avg']
        result.ligand_rmsd_std = rmsd_data['ligand_rmsd_std']
        result.protein_rmsd_avg = rmsd_data['protein_rmsd_avg']
        result.protein_rmsd_std = rmsd_data['protein_rmsd_std']
        
        print(f"    Ligand RMSD: {result.ligand_rmsd_avg:.3f} ± {result.ligand_rmsd_std:.3f} Å")
        print(f"    Protein RMSD: {result.protein_rmsd_avg:.3f} ± {result.protein_rmsd_std:.3f} Å")
        
        # Calculate scores
        result = calculate_scores(result)
        print(f"    Binding score: {result.binding_score:.4f}")
        print(f"    Stability score: {result.stability_score:.4f}")
        print(f"    Overall score: {result.overall_score:.4f}")
        
        results.append(result)
    
    # Sort by overall score
    results.sort(key=lambda x: x.overall_score, reverse=True)
    
    # Create results table
    print(f"\n{'='*80}")
    print(f"FINAL RANKINGS WITH RMSD ANALYSIS")
    print(f"{'='*80}")
    print(f"{'Rank':<4} {'Molecule':<12} {'Overall':<8} {'Binding':<8} {'Stability':<9} {'Avg Energy':<12} {'Ligand RMSD':<12} {'Protein RMSD'}")
    print("-" * 80)
    
    for i, result in enumerate(results, 1):
        print(f"{i:<4} {result.molecule_name:<12} {result.overall_score:<8.4f} {result.binding_score:<8.4f} "
              f"{result.stability_score:<9.4f} {result.average_energy:<12.1f} {result.ligand_rmsd_avg:<12.3f} {result.protein_rmsd_avg:<12.3f}")
    
    # Save to CSV
    csv_data = []
    for i, result in enumerate(results, 1):
        csv_data.append({
            'Rank': i,
            'Molecule_Name': result.molecule_name,
            'Overall_Score': result.overall_score,
            'Binding_Score': result.binding_score,
            'Stability_Score': result.stability_score,
            'Average_Energy_kJ_mol': result.average_energy,
            'Energy_Std_kJ_mol': result.energy_std,
            'Ligand_RMSD_Avg_A': result.ligand_rmsd_avg,
            'Ligand_RMSD_Std_A': result.ligand_rmsd_std,
            'Protein_RMSD_Avg_A': result.protein_rmsd_avg,
            'Protein_RMSD_Std_A': result.protein_rmsd_std
        })
    
    df = pd.DataFrame(csv_data)
    output_file = results_dir / "final_rankings_with_rmsd.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
