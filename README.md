# Easy-MD High-Throughput Drug Screening Pipeline

**🎯 Automated molecular dynamics simulations for drug discovery with GPU acceleration**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![OpenMM](https://img.shields.io/badge/OpenMM-CUDA-green.svg)](https://openmm.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🚀 Quick Start

```bash
# 1. Setup environment
conda env create -f environment.yaml
conda activate easymd

# 2. Prepare your data
# - Place protein PDB file in protien/ directory
# - Place ligand SDF files in molecules/ directory

# 3. IMPORTANT: Edit protein file path
# Open run_high_throughput.py and update line 481:
# protein_file = "protien/your_actual_protein_name.pdb"

# 4. Run high-throughput MD simulations
python run_high_throughput.py

# 5. Analyze results with RMSD
python enhanced_analysis_with_rmsd.py

# 6. Generate energy plots
python plot_energy_trajectories.py
```

## 🎯 What This Pipeline Does

- **Automated MD Workflow**: Complete simulation pipeline from setup to analysis
- **High-Throughput Processing**: Batch processing of multiple ligands against target protein
- **GPU Acceleration**: CUDA-optimized for fast execution
- **Comprehensive Analysis**: Binding energy + structural stability (RMSD) analysis
- **Publication-Ready Results**: Ranked CSV files and publication-quality plots

## 📁 File Structure

```
easy-md/
├── run_high_throughput.py          # Main simulation runner
├── enhanced_analysis_with_rmsd.py  # RMSD analysis script  
├── plot_energy_trajectories.py     # Visualization generator
├── environment.yaml                # Conda environment
├── molecules/                      # Input ligands (SDF files)
├── protien/                        # Target protein (PDB file)
├── src/easy_md/                    # Core pipeline modules
└── md_screening_results/           # Output directory (created automatically)
```

## 🛠️ Requirements

- **Python 3.9+** with conda/mamba
- **GPU recommended** (CUDA) for fast execution
- **8-16GB RAM** and **10GB+ disk space**
- **Dependencies**: OpenMM, MDAnalysis, OpenFF, RDKit, NumPy, Pandas

## � Input Files

### Protein Structure
- Place your target protein PDB file in `protien/` directory
- Ensure proper structure (single chain recommended for docking)

### Ligand Library  
- Place ligand SDF files in `molecules/` directory
- Supports multiple molecules for batch processing
- Files should be properly protonated and energy minimized

## 🔧 Configuration

**IMPORTANT**: Before running simulations, edit `run_high_throughput.py` to customize:

```python
# Simulation parameters
md_steps = 5000          # Number of MD steps
platform = "CUDA"       # Use "CPU" if no GPU available

# UPDATE THIS PATH to match your protein file name
protein_file = "protien/your_actual_protein_name.pdb"  # ⚠️ CHANGE THIS!
molecules_dir = "molecules"
```

**⚠️ Critical Step**: Update the `protein_file` path on line 481 to match your actual protein PDB filename in the `protien/` directory.

## � Output Files

The pipeline generates:

- **Energy trajectories**: `md_id_0.log` for each molecule
- **Trajectory files**: `md_trajetory_id_0.dcd` for structural analysis
- **Rankings**: `final_rankings_with_rmsd.csv` with comprehensive scores
- **Visualizations**: Energy plots and RMSD analysis charts

## 🧪 Analysis Metrics

### Binding Affinity
- **Potential energy** throughout simulation
- **Energy convergence** analysis
- **Average binding energy** calculation

### Structural Stability
- **Ligand RMSD** - binding pose stability
- **Protein RMSD** - target structure integrity
- **Composite scoring** for robust ranking

## 🚀 Performance

- **Runtime**: 1-2 hours per molecule (GPU) vs 6-12 hours (CPU)
- **Scalability**: Easily adaptable to larger molecule libraries
- **Accuracy**: OpenMM + OpenFF force field with explicit solvent

## 🔬 Scientific Applications

This pipeline is suitable for:

- **Drug discovery** - virtual screening of compound libraries
- **Lead optimization** - comparing molecular variants
- **Target analysis** - studying protein-ligand interactions
- **Academic research** - educational MD simulations

## 📚 Getting Started

1. **Clone the repository**
2. **Set up the conda environment**: `conda env create -f environment.yaml`
3. **Prepare your input files** (protein PDB and ligand SDF files)
4. **Run the pipeline**: `python run_high_throughput.py`
5. **Analyze results**: `python enhanced_analysis_with_rmsd.py`

## 🤝 Contributing

This is an open-source pipeline for molecular dynamics drug screening. Contributions, issues, and feature requests are welcome.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
