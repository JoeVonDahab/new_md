#!/usr/bin/env python3
"""
Visualize energy trajectories from MD simulations to assess convergence.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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

def create_energy_plots():
    """Create energy trajectory plots for all molecules."""
    results_dir = Path("md_screening_results")
    
    molecules = ["Crizotinib", "Axitinib", "Vandetanib", "Risdiplam"]
    colors = ['blue', 'red', 'green', 'orange']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Energy Trajectories - MD Simulations', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for i, molecule in enumerate(molecules):
        log_file = results_dir / molecule / "output" / "md_id_0.log"
        
        if log_file.exists():
            times, energies = parse_energy_log(log_file)
            
            if times and energies:
                times = np.array(times)
                energies = np.array(energies)
                
                # Plot energy trajectory
                axes[i].plot(times, energies, color=colors[i], alpha=0.7, linewidth=1)
                
                # Add running average
                window = max(5, len(energies) // 20)
                if len(energies) > window:
                    running_avg = np.convolve(energies, np.ones(window)/window, mode='valid')
                    running_times = times[window-1:]
                    axes[i].plot(running_times, running_avg, 
                               color='black', linewidth=2, label='Running Average')
                
                # Calculate statistics
                avg_energy = np.mean(energies)
                final_energy = energies[-1]
                
                axes[i].axhline(y=avg_energy, color='red', linestyle='--', 
                              alpha=0.8, label=f'Average: {avg_energy:.0f}')
                
                axes[i].set_title(f'{molecule}\nFinal: {final_energy:.0f} kJ/mol', 
                                fontweight='bold')
                axes[i].set_xlabel('Time (ps)')
                axes[i].set_ylabel('Potential Energy (kJ/mol)')
                axes[i].grid(True, alpha=0.3)
                axes[i].legend(fontsize=8)
                
                # Set y-axis to show interesting range
                y_range = np.max(energies) - np.min(energies)
                y_center = np.mean(energies)
                axes[i].set_ylim(y_center - y_range*0.6, y_center + y_range*0.6)
        else:
            axes[i].text(0.5, 0.5, f'No data for {molecule}', 
                       ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(molecule)
    
    plt.tight_layout()
    
    # Save plot
    output_file = results_dir / "energy_trajectories.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Energy trajectory plots saved to: {output_file}")
    
    # Create comparison plot
    create_comparison_plot(results_dir, molecules, colors)

def create_comparison_plot(results_dir, molecules, colors):
    """Create a comparison plot of all energy trajectories."""
    plt.figure(figsize=(12, 8))
    
    final_energies = []
    avg_energies = []
    
    for i, molecule in enumerate(molecules):
        log_file = results_dir / molecule / "output" / "md_id_0.log"
        
        if log_file.exists():
            times, energies = parse_energy_log(log_file)
            
            if times and energies:
                times = np.array(times)
                energies = np.array(energies)
                
                # Plot with offset for visibility
                offset = i * 1000  # 1000 kJ/mol offset between molecules
                plt.plot(times, energies + offset, color=colors[i], 
                        alpha=0.7, linewidth=1.5, label=f'{molecule} (offset +{offset})')
                
                final_energies.append(energies[-1])
                avg_energies.append(np.mean(energies))
    
    plt.xlabel('Time (ps)')
    plt.ylabel('Potential Energy (kJ/mol) + Offset')
    plt.title('Energy Trajectories Comparison (with offsets for visibility)', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = results_dir / "energy_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Energy comparison plot saved to: {output_file}")
    
    # Print summary
    print("\nEnergy Summary:")
    print("=" * 50)
    for i, molecule in enumerate(molecules):
        if i < len(final_energies):
            print(f"{molecule:12}: Final = {final_energies[i]:8.1f}, Average = {avg_energies[i]:8.1f} kJ/mol")

if __name__ == "__main__":
    try:
        create_energy_plots()
    except ImportError:
        print("Error: matplotlib not available. Please install with: pip install matplotlib")
    except Exception as e:
        print(f"Error creating plots: {e}")
