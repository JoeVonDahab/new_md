#!/usr/bin/env python3
"""
Visualize energy trajectories, binding affinity, and RMSD metrics from MD simulations.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import MDAnalysis as mda
from MDAnalysis.analysis import rms

def get_all_molecules(results_dir):
    """Get all molecule directories."""
    if not results_dir.exists():
        return []
    
    molecule_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    molecules = [d.name for d in molecule_dirs]
    return sorted(molecules)

def parse_energy_log(log_file):
    """Parse energy trajectory from log file."""
    times = []
    energies = []
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            
        # Skip header
        for line in lines[1:]:
            if line.strip():
                parts = line.split('\t')
                if len(parts) >= 4:
                    try:
                        time = float(parts[2])  # Time in ps
                        energy = float(parts[3])  # Energy in kJ/mol
                        times.append(time)
                        energies.append(energy)
                    except:
                        continue
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
    
    return times, energies

def analyze_rmsd(mol_dir):
    """Analyze RMSD for a molecule."""
    structure_file = mol_dir / "output" / "openff_interchange.pdb"
    traj_file = mol_dir / "output" / "md_trajetory_id_0.dcd"
    
    results = {'ligand_rmsd_avg': 0.0, 'protein_rmsd_avg': 0.0}
    
    if not (structure_file.exists() and traj_file.exists()):
        return results
    
    try:
        import MDAnalysis as mda
        from MDAnalysis.analysis import rms
        
        u = mda.Universe(str(structure_file), str(traj_file))
        
        # Protein analysis
        protein = u.select_atoms("protein and backbone")
        if len(protein) > 0:
            R_protein = rms.RMSD(protein, ref=protein)
            R_protein.run()
            results['protein_rmsd_avg'] = np.mean(R_protein.results.rmsd[:, 2])
        
        # Ligand analysis
        ligand = u.select_atoms("resname UNK")
        if len(ligand) == 0:
            ligand = u.select_atoms("not protein and not resname HOH and not resname Na+ and not resname Cl-")
        
        if len(ligand) > 0:
            R_ligand = rms.RMSD(ligand, ref=ligand)
            R_ligand.run()
            results['ligand_rmsd_avg'] = np.mean(R_ligand.results.rmsd[:, 2])
    
    except Exception as e:
        print(f"RMSD analysis error for {mol_dir.name}: {e}")
    
    return results

def create_energy_comparison_plot():
    """Create a single plot comparing all energy trajectories."""
    results_dir = Path("md_screening_results")
    molecules = get_all_molecules(results_dir)
    
    if not molecules:
        print("No molecules found in md_screening_results!")
        return
    
    plt.figure(figsize=(15, 10))
    
    # Generate colors for all molecules
    colors = plt.cm.tab20(np.linspace(0, 1, len(molecules)))
    
    for i, molecule in enumerate(molecules):
        log_file = results_dir / molecule / "output" / "md_id_0.log"
        
        if log_file.exists():
            times, energies = parse_energy_log(log_file)
            
            if times and energies:
                times = np.array(times)
                energies = np.array(energies)
                
                # Plot energy trajectory
                plt.plot(times, energies, color=colors[i], 
                        alpha=0.7, linewidth=1.5, label=molecule)
    
    plt.xlabel('Time (ps)', fontsize=12)
    plt.ylabel('Potential Energy (kJ/mol)', fontsize=12)
    plt.title('Energy Trajectories Comparison - All Molecules', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Create legend with smaller font and multiple columns
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    
    plt.tight_layout()
    
    output_file = results_dir / "energy_trajectories_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Energy comparison plot saved to: {output_file}")
    plt.close()

def create_binding_affinity_plot():
    """Create a binding affinity comparison plot."""
    results_dir = Path("md_screening_results")
    molecules = get_all_molecules(results_dir)
    
    if not molecules:
        print("No molecules found!")
        return
    
    molecule_names = []
    avg_energies = []
    
    for molecule in molecules:
        log_file = results_dir / molecule / "output" / "md_id_0.log"
        
        if log_file.exists():
            times, energies = parse_energy_log(log_file)
            
            if energies:
                molecule_names.append(molecule)
                avg_energies.append(np.mean(energies))
    
    if not molecule_names:
        print("No energy data found!")
        return
    
    # Sort by binding affinity (more negative = better binding)
    sorted_data = sorted(zip(molecule_names, avg_energies), key=lambda x: x[1])
    molecule_names, avg_energies = zip(*sorted_data)
    
    plt.figure(figsize=(15, 10))
    bars = plt.bar(range(len(molecule_names)), avg_energies, 
                   color=plt.cm.viridis(np.linspace(0, 1, len(molecule_names))))
    
    plt.xlabel('Molecules', fontsize=12)
    plt.ylabel('Average Potential Energy (kJ/mol)', fontsize=12)
    plt.title('Binding Affinity Comparison (Lower = Better Binding)', fontsize=14, fontweight='bold')
    
    # Rotate x-axis labels
    plt.xticks(range(len(molecule_names)), molecule_names, rotation=45, ha='right', fontsize=8)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}', ha='center', va='bottom', fontsize=6)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    output_file = results_dir / "binding_affinity_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Binding affinity plot saved to: {output_file}")
    plt.close()

def create_rmsd_comparison_plot():
    """Create RMSD comparison plots."""
    results_dir = Path("md_screening_results")
    molecules = get_all_molecules(results_dir)
    
    if not molecules:
        print("No molecules found!")
        return
    
    molecule_names = []
    ligand_rmsds = []
    protein_rmsds = []
    
    print("Analyzing RMSD for all molecules...")
    
    for molecule in molecules:
        mol_dir = results_dir / molecule
        rmsd_data = analyze_rmsd(mol_dir)
        
        if rmsd_data['ligand_rmsd_avg'] > 0 or rmsd_data['protein_rmsd_avg'] > 0:
            molecule_names.append(molecule)
            ligand_rmsds.append(rmsd_data['ligand_rmsd_avg'])
            protein_rmsds.append(rmsd_data['protein_rmsd_avg'])
    
    if not molecule_names:
        print("No RMSD data found!")
        return
    
    # Create subplot for ligand and protein RMSD
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Ligand RMSD plot
    bars1 = ax1.bar(range(len(molecule_names)), ligand_rmsds, 
                    color=plt.cm.plasma(np.linspace(0, 1, len(molecule_names))))
    ax1.set_xlabel('Molecules', fontsize=12)
    ax1.set_ylabel('Ligand RMSD (Å)', fontsize=12)
    ax1.set_title('Ligand RMSD Comparison (Lower = More Stable)', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(molecule_names)))
    ax1.set_xticklabels(molecule_names, rotation=45, ha='right', fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=6)
    
    # Protein RMSD plot
    bars2 = ax2.bar(range(len(molecule_names)), protein_rmsds, 
                    color=plt.cm.cividis(np.linspace(0, 1, len(molecule_names))))
    ax2.set_xlabel('Molecules', fontsize=12)
    ax2.set_ylabel('Protein RMSD (Å)', fontsize=12)
    ax2.set_title('Protein RMSD Comparison (Lower = More Stable)', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(molecule_names)))
    ax2.set_xticklabels(molecule_names, rotation=45, ha='right', fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=6)
    
    plt.tight_layout()
    
    output_file = results_dir / "rmsd_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"RMSD comparison plot saved to: {output_file}")
    plt.close()

def main():
    """Create all comparison plots."""
    print("Creating comprehensive analysis plots...")
    
    try:
        create_energy_comparison_plot()
        create_binding_affinity_plot()
        create_rmsd_comparison_plot()
        print("\nAll plots created successfully!")
        
    except ImportError as e:
        if "matplotlib" in str(e):
            print("Error: matplotlib not available. Please install with: pip install matplotlib")
        elif "MDAnalysis" in str(e):
            print("Warning: MDAnalysis not available. RMSD plots will be skipped.")
            print("To enable RMSD analysis, install with: pip install MDAnalysis")
            create_energy_comparison_plot()
            create_binding_affinity_plot()
        else:
            print(f"Import error: {e}")
    except Exception as e:
        print(f"Error creating plots: {e}")

if __name__ == "__main__":
    main()
