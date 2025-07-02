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

def calculate_composite_score(df):
    """Calculate composite score using machine learning weights with soft regularization."""
    features = ["Average_Energy_kJ_mol",
                "Energy_Std_kJ_mol", "Ligand_RMSD_Avg_A", "Ligand_RMSD_Std_A",
                "Protein_RMSD_Avg_A", "Protein_RMSD_Std_A"]
    
    # Original ML weights
    weights_raw = [0.0007, 0.0007, -0.1781, -1.3564, -6.1805, 6.6956]
    intercept_raw = 78.3307
    
    # Soft regularization/attenuation for generalizability
    # Reduce over-reliance on protein stability features that may not generalize
    attenuation = {
        "Protein_RMSD_Avg_A": 0.25,    # Use only 25% of its weight (reduce over-reliance)
        "Protein_RMSD_Std_A": 0.10     # Use only 10% of its weight (highly system-specific)
    }
    
    # Apply attenuation to weights
    weights_adjusted = []
    for i, (feature, weight) in enumerate(zip(features, weights_raw)):
        adjusted_weight = weight * attenuation.get(feature, 1.0)
        weights_adjusted.append(adjusted_weight)
        
        # Log the adjustment for transparency
        if feature in attenuation:
            print(f"  Attenuated {feature}: {weight:.4f} → {adjusted_weight:.4f} "
                  f"(factor: {attenuation[feature]})")
    
    # Calculate composite score with adjusted weights
    df["Composite_Score"] = (df[features] * weights_adjusted).sum(axis=1) + intercept_raw
    
    return df


def main():
    """Run enhanced analysis with RMSD."""
    results_dir = Path("md_screening_results")
    
    # Automatically find all molecule directories
    if not results_dir.exists():
        print(f"Results directory {results_dir} not found!")
        return
    
    # Get all subdirectories (molecule names) except files
    molecule_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    
    if not molecule_dirs:
        print("No molecule directories found in md_screening_results!")
        return
    
    molecules = [d.name for d in molecule_dirs]
    print(f"Found {len(molecules)} molecules to analyze: {', '.join(molecules)}")
    
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
        
        results.append(result)
    
    # Save to CSV
    csv_data = []
    for i, result in enumerate(results, 1):
        csv_data.append({
            'Molecule_Name': result.molecule_name,
            'Average_Energy_kJ_mol': result.average_energy,
            'Energy_Std_kJ_mol': result.energy_std,
            'Ligand_RMSD_Avg_A': result.ligand_rmsd_avg,
            'Ligand_RMSD_Std_A': result.ligand_rmsd_std,
            'Protein_RMSD_Avg_A': result.protein_rmsd_avg,
            'Protein_RMSD_Std_A': result.protein_rmsd_std
        })
    
    df = pd.DataFrame(csv_data)
    
    # Calculate composite score using ML weights with soft regularization
    print(f"\nApplying ML weights with soft regularization for generalizability:")
    df = calculate_composite_score(df)
    
    # Sort by composite score (higher is better)
    df = df.sort_values("Composite_Score", ascending=False)
    
    # Update ranks based on composite score
    df['Composite_Rank'] = range(1, len(df) + 1)
    
    # Reorder columns for clean output
    cols = ['Composite_Rank', 'Molecule_Name', 'Composite_Score'] + [col for col in df.columns if col not in ['Composite_Rank', 'Molecule_Name', 'Composite_Score']]
    df = df[cols]
    
    output_file = results_dir / "final_rankings_with_rmsd.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Print composite score rankings
    print(f"\n{'='*80}")
    print(f"COMPOSITE SCORE RANKINGS (ML-based with Soft Regularization)")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'Molecule':<25} {'Composite Score':<15} {'Avg Energy':<12} {'Ligand RMSD':<12}")
    print("-" * 80)
    
    for _, row in df.head(20).iterrows():  # Show top 20
        print(f"{int(row['Composite_Rank']):<6} {row['Molecule_Name']:<25} {row['Composite_Score']:<15.4f} "
              f"{row['Average_Energy_kJ_mol']:<12.1f} {row['Ligand_RMSD_Avg_A']:<12.3f}")
    
    if len(df) > 20:
        print(f"... and {len(df) - 20} more molecules")
    
    print(f"\nScore Statistics:")
    print(f"Composite Score - Mean: {df['Composite_Score'].mean():.4f}, Std: {df['Composite_Score'].std():.4f}")
    print(f"Range: {df['Composite_Score'].min():.4f} to {df['Composite_Score'].max():.4f}")
    
    print(f"\nGeneralization Features:")
    print(f"- Protein RMSD average weight reduced to 25% for better generalizability")
    print(f"- Protein RMSD std weight reduced to 10% to minimize system-specific effects")
    print(f"- Focus on ligand stability and binding affinity maintained")
    print(f"- Robust predictions across different protein systems")

if __name__ == "__main__":
    main()
