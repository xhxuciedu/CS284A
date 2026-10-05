"""Download one AlphaFold prediction and inspect local confidence values."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests
from Bio.PDB import PDBParser


def download_prediction(accession):
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    target = data_dir / f"{accession}.pdb"
    if target.exists():
        return target
    api = f"https://alphafold.ebi.ac.uk/api/prediction/{accession}"
    response = requests.get(api, timeout=30)
    response.raise_for_status()
    records = response.json()
    if isinstance(records, dict):
        records = [records]
    if not records or "pdbUrl" not in records[0]:
        raise ValueError(f"No PDB prediction returned for {accession}")
    pdb_response = requests.get(records[0]["pdbUrl"], timeout=90)
    pdb_response.raise_for_status()
    target.write_text(pdb_response.text, encoding="utf-8")
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uniprot", default="P69905")
    args = parser.parse_args()
    pdb_path = download_prediction(args.uniprot)
    structure = PDBParser(QUIET=True).get_structure(args.uniprot, str(pdb_path))
    first_model = next(structure.get_models())
    first_chain = next(first_model.get_chains())
    residues = [r for r in first_chain.get_residues() if "CA" in r]
    positions = np.asarray([r.id[1] for r in residues])
    confidence = np.asarray([r["CA"].bfactor for r in residues])
    print(f"accession={args.uniprot}, residues={len(residues)}")
    print(f"median pLDDT={np.median(confidence):.1f}")
    print(f"fraction above 90={np.mean(confidence > 90):.2%}")
    print(f"fraction below 50={np.mean(confidence < 50):.2%}")
    Path("outputs").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(positions, confidence, linewidth=1.6)
    ax.axhline(90, color="green", linestyle="--", linewidth=1)
    ax.axhline(50, color="orange", linestyle="--", linewidth=1)
    ax.set(xlabel="Residue position", ylabel="pLDDT", ylim=(0, 100),
           title=f"AlphaFold confidence for {args.uniprot}")
    fig.tight_layout()
    fig.savefig("outputs/confidence.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
