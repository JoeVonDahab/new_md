# Input Ligands

Place your ligand SDF files here for batch processing.

**Requirements:**
- SDF format (.sdf extension)
- Properly protonated structures
- 3D coordinates (energy minimized recommended)
- One molecule per file

**Example filenames:**
- `compound_001.sdf`
- `ligand_A.sdf`
- `drug_candidate.sdf`

**Notes:**
- The pipeline will automatically process all `.sdf` files in this directory
- Molecule names are extracted from filenames
- Avoid special characters in filenames
