# Input Files

## Protein Structure

Place your target protein PDB file here.

**Requirements:**
- Clean PDB structure (single chain recommended)
- Proper formatting and valid coordinates
- Remove water molecules and heteroatoms (except target ligands)

**Example filename:** `your_protein.pdb`

**Update the filename in `run_high_throughput.py`:**
```python
protein_file = "protien/your_protein.pdb"
```
