"""
Open-source structure relaxation and interface scoring utilities.

This backend intentionally preserves the small API used by ``filter_utils`` so
Germinal can run without PyRosetta. The scores are not Rosetta-equivalent:
metrics such as ``interface_sc``, ``interface_packstat`` and ``sap_score`` are
geometry/SASA proxies and should be calibrated separately from Rosetta filters.
"""

import os
import shutil
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import mdtraj as md
import numpy as np
from Bio.PDB import PDBIO, PDBParser, Superimposer
from Bio.PDB.Polypeptide import is_aa


AA3_TO_1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}
HYDROPHOBIC_1 = set("ACFILMPVWY")
HYDROPHOBIC_3 = {"ALA", "CYS", "PHE", "ILE", "LEU", "MET", "PRO", "VAL", "TRP", "TYR"}
DONOR_ACCEPTOR = {"N", "O", "S"}


def clean_pdb(pdb_file):
    with open(pdb_file, "r") as handle:
        relevant_lines = [
            line
            for line in handle
            if line.startswith(("ATOM", "HETATM", "MODEL", "TER", "END"))
        ]
    with open(pdb_file, "w") as handle:
        handle.writelines(relevant_lines)


def hotspot_residues(
    trajectory_pdb, binder_chain="B", atom_distance_cutoff=4.0, target_chain="A"
):
    target_chain = target_chain.split(",") if isinstance(target_chain, str) else target_chain
    structure = _parse_structure(trajectory_pdb)
    model = _model(structure)
    if len(list(model.get_chains())) == 2:
        binder_chain = "B"
    binder_residues = _chain_residues(structure, binder_chain)
    target_residues = []
    for chain_id in target_chain:
        target_residues.extend(_chain_residues(structure, chain_id))

    interacting = {}
    target_atoms = [atom for res in target_residues for atom in _heavy_atoms(res)]
    for binder_res in binder_residues:
        for binder_atom in _heavy_atoms(binder_res):
            if any(_atom_distance(binder_atom, target_atom) <= atom_distance_cutoff for target_atom in target_atoms):
                aa = AA3_TO_1.get(binder_res.resname)
                if aa:
                    interacting[_residue_number(binder_res)] = aa
                break
    return interacting


def _parse_structure(pdb_path):
    parser = PDBParser(QUIET=True)
    return parser.get_structure("germinal", pdb_path)


def _model(structure):
    return next(structure.get_models())


def _chain_residues(structure, chain_id: str):
    chain = _model(structure)[chain_id]
    return [res for res in chain if is_aa(res, standard=True)]


def _residue_number(residue) -> int:
    return int(residue.id[1])


def _heavy_atoms(residue):
    return [atom for atom in residue if atom.element != "H"]


def _residue_center(residue) -> np.ndarray:
    if "CB" in residue:
        return residue["CB"].coord
    if "CA" in residue:
        return residue["CA"].coord
    atoms = _heavy_atoms(residue)
    return np.mean([a.coord for a in atoms], axis=0)


def _atom_distance(a, b) -> float:
    return float(np.linalg.norm(a.coord - b.coord))


def _copy_or_clean(input_pdb: str, output_pdb: str) -> None:
    if input_pdb != output_pdb:
        shutil.copyfile(input_pdb, output_pdb)
    clean_pdb(output_pdb)


def _mdtraj_chain_atom_indices(traj, chain_ids: Sequence[str]) -> List[int]:
    wanted = set(chain_ids)
    return [
        atom.index
        for atom in traj.topology.atoms
        if atom.residue.chain.chain_id in wanted
    ]


def _sasa_by_atom(pdb_path: str, chain_ids: Optional[Sequence[str]] = None) -> float:
    traj = md.load(pdb_path)
    if chain_ids is not None:
        atom_indices = _mdtraj_chain_atom_indices(traj, chain_ids)
        if not atom_indices:
            return 0.0
        traj = traj.atom_slice(atom_indices)
    sasa_nm2 = md.shrake_rupley(traj, mode="atom")[0]
    return float(np.sum(sasa_nm2) * 100.0)


def _residue_sasa(pdb_path: str, chain_id: Optional[str] = None) -> Dict[Tuple[str, int], float]:
    traj = md.load(pdb_path)
    sasa_nm2 = md.shrake_rupley(traj, mode="residue")[0]
    result = {}
    for res, sasa in zip(traj.topology.residues, sasa_nm2):
        if chain_id is not None and res.chain.chain_id != chain_id:
            continue
        result[(res.chain.chain_id, int(res.resSeq))] = float(sasa * 100.0)
    return result


def _openmm_energy(pdb_path: str) -> Optional[float]:
    try:
        from openmm import unit
        from openmm.app import ForceField, Modeller, NoCutoff, PDBFile, Simulation
        from openmm import LangevinIntegrator
    except Exception:
        return None

    try:
        pdb = PDBFile(pdb_path)
        forcefield = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
        modeller = Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(forcefield)
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=NoCutoff,
            constraints=None,
        )
        integrator = LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
        simulation = Simulation(modeller.topology, system, integrator)
        simulation.context.setPositions(modeller.positions)
        state = simulation.context.getState(getEnergy=True)
        return float(state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))
    except Exception:
        return None


def _interface_contacts(structure, chain1: str, chain2: str, cutoff: float = 5.0):
    residues1 = _chain_residues(structure, chain1)
    residues2 = _chain_residues(structure, chain2)
    contacts = []
    for res1 in residues1:
        atoms1 = _heavy_atoms(res1)
        if not atoms1:
            continue
        for res2 in residues2:
            atoms2 = _heavy_atoms(res2)
            if not atoms2:
                continue
            min_dist = min(_atom_distance(a, b) for a in atoms1 for b in atoms2)
            if min_dist <= cutoff:
                contacts.append((res1, res2, min_dist))
    return contacts


def _hbond_count(contacts) -> int:
    hbonds = 0
    for res1, res2, _ in contacts:
        atoms1 = [a for a in _heavy_atoms(res1) if a.element in DONOR_ACCEPTOR]
        atoms2 = [a for a in _heavy_atoms(res2) if a.element in DONOR_ACCEPTOR]
        if any(_atom_distance(a, b) <= 3.5 for a in atoms1 for b in atoms2):
            hbonds += 1
    return hbonds


def _contact_area_proxy(contacts) -> float:
    if not contacts:
        return 0.0
    # Smoothly reward many close residue contacts. This is a proxy, not Sc.
    values = [max(0.0, 5.0 - dist) / 5.0 for _, _, dist in contacts]
    return float(np.clip(np.mean(values) + 0.35 * np.log1p(len(contacts)) / 5.0, 0.0, 1.0))


def pr_relax(pdb_file, relaxed_pdb_path):
    """OpenMM restrained minimization if available; otherwise clean-copy PDB."""
    if os.path.exists(relaxed_pdb_path):
        return
    try:
        from openmm import CustomExternalForce, LangevinIntegrator, unit
        from openmm.app import ForceField, Modeller, NoCutoff, PDBFile, Simulation
    except Exception:
        _copy_or_clean(pdb_file, relaxed_pdb_path)
        return

    try:
        pdb = PDBFile(pdb_file)
        forcefield = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
        modeller = Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(forcefield)
        system = forcefield.createSystem(modeller.topology, nonbondedMethod=NoCutoff, constraints=None)

        restraint = CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        restraint.addGlobalParameter("k", 5.0 * unit.kilocalories_per_mole / unit.angstroms**2)
        restraint.addPerParticleParameter("x0")
        restraint.addPerParticleParameter("y0")
        restraint.addPerParticleParameter("z0")
        for idx, pos in enumerate(modeller.positions):
            restraint.addParticle(idx, pos.value_in_unit(unit.nanometer))
        system.addForce(restraint)

        integrator = LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
        simulation = Simulation(modeller.topology, system, integrator)
        simulation.context.setPositions(modeller.positions)
        simulation.minimizeEnergy(maxIterations=500)
        state = simulation.context.getState(getPositions=True)
        with open(relaxed_pdb_path, "w") as handle:
            PDBFile.writeFile(modeller.topology, state.getPositions(), handle)
        clean_pdb(relaxed_pdb_path)
    except Exception:
        _copy_or_clean(pdb_file, relaxed_pdb_path)


def pr_relax_parallel(pdb_file, output_dir, design_name, dalphaball_path=None, n_relax=5):
    relaxed_paths = []
    for i in range(n_relax):
        path = os.path.join(output_dir, f"{design_name}_relaxed_{i}.pdb")
        pr_relax(pdb_file, path)
        if os.path.exists(path):
            relaxed_paths.append(path)
    return relaxed_paths


def score_interface(pdb_file, binder_chain="B", target_chain="A"):
    structure = _parse_structure(pdb_file)
    target_chains = target_chain.split(",") if isinstance(target_chain, str) else list(target_chain)
    contacts = []
    for ch in target_chains:
        contacts.extend(_interface_contacts(structure, ch, binder_chain, cutoff=5.0))

    interface_AA = {aa: 0 for aa in "ACDEFGHIKLMNPQRSTVWY"}
    interface_residues_set = hotspot_residues(pdb_file, binder_chain, target_chain=target_chain)
    interface_residues_pdb_ids = []
    for pdb_res_num, aa_type in interface_residues_set.items():
        if aa_type in interface_AA:
            interface_AA[aa_type] += 1
        interface_residues_pdb_ids.append(f"{binder_chain}{pdb_res_num}")
    interface_nres = len(interface_residues_pdb_ids)
    interface_residues_pdb_ids_str = ",".join(interface_residues_pdb_ids)

    hydrophobic_count = sum(interface_AA[aa] for aa in HYDROPHOBIC_1)
    interface_hydrophobicity = (hydrophobic_count / interface_nres) * 100 if interface_nres else 0
    interface_hbonds = _hbond_count(contacts)

    complex_sasa = _sasa_by_atom(pdb_file, [binder_chain] + target_chains)
    binder_sasa = _sasa_by_atom(pdb_file, [binder_chain])
    target_sasa = sum(_sasa_by_atom(pdb_file, [ch]) for ch in target_chains)
    interface_dSASA = max(0.0, binder_sasa + target_sasa - complex_sasa)
    interface_fraction = (interface_dSASA / binder_sasa) * 100 if binder_sasa else 0

    residue_sasa = _residue_sasa(pdb_file, binder_chain)
    binder_residues = _chain_residues(structure, binder_chain)
    exposed = [
        res
        for res in binder_residues
        if residue_sasa.get((binder_chain, _residue_number(res)), 0.0) > 20.0
    ]
    surface_hydrophobicity = (
        sum(res.resname in HYDROPHOBIC_3 for res in exposed) / len(exposed)
        if exposed
        else 0
    )

    binder_energy = _openmm_energy(pdb_file)
    if binder_energy is None or not np.isfinite(binder_energy) or abs(binder_energy) > 1e6:
        # Lightweight deterministic proxy: clashes/contacts lower the score.
        binder_energy = -float(len(contacts))

    interface_sc = _contact_area_proxy(contacts)
    interface_packstat = min(1.0, interface_dSASA / max(1.0, 25.0 * interface_nres))
    interface_dG = -0.015 * interface_dSASA - 0.4 * interface_hbonds
    interface_dG_SASA_ratio = (interface_dG / interface_dSASA) * 100 if interface_dSASA else 0
    unsat_proxy = max(0, interface_nres - interface_hbonds)
    hbond_pct = (interface_hbonds / interface_nres) * 100 if interface_nres else None
    unsat_pct = (unsat_proxy / interface_nres) * 100 if interface_nres else None

    scores = {
        "binder_score": binder_energy,
        "surface_hydrophobicity": surface_hydrophobicity,
        "interface_sc": interface_sc,
        "interface_loop_sc": interface_sc,
        "interface_loop_sc_area": interface_dSASA,
        "interface_packstat": interface_packstat,
        "interface_dG": interface_dG,
        "interface_dSASA": interface_dSASA,
        "interface_dG_SASA_ratio": interface_dG_SASA_ratio,
        "interface_fraction": interface_fraction,
        "interface_hydrophobicity": interface_hydrophobicity,
        "interface_nres": interface_nres,
        "interface_interface_hbonds": interface_hbonds,
        "interface_hbond_percentage": hbond_pct,
        "interface_delta_unsat_hbonds": unsat_proxy,
        "interface_delta_unsat_hbonds_percentage": unsat_pct,
    }
    scores = {k: round(v, 2) if isinstance(v, float) else v for k, v in scores.items()}
    return scores, interface_AA, interface_residues_pdb_ids_str


def score_interface_ensemble(relaxed_pdb_paths, binder_chain="B", target_chain="A", score_mode="average"):
    all_scores, all_aa, all_residues = [], [], []
    for pdb_path in relaxed_pdb_paths:
        scores, aa, residues = score_interface(pdb_path, binder_chain, target_chain)
        all_scores.append(scores)
        all_aa.append(aa)
        all_residues.append(residues)
    if not all_scores:
        raise RuntimeError("All open-source score_interface calls failed")
    best_idx = int(np.argmin([s["binder_score"] for s in all_scores]))
    if score_mode == "best":
        return all_scores[best_idx], all_aa[best_idx], all_residues[best_idx], relaxed_pdb_paths[best_idx]
    result_scores = {}
    for key in all_scores[0]:
        values = [s[key] for s in all_scores if s.get(key) is not None]
        result_scores[key] = float(np.mean(values)) if values and isinstance(values[0], (int, float)) else all_scores[0][key]
    return result_scores, all_aa[best_idx], all_residues[best_idx], relaxed_pdb_paths[best_idx]


def get_sap_score(
    pdb,
    binder_chain=None,
    only_binder=False,
    hydrophobic_aa=None,
    patch_radius=8,
    limit_sasa=1,
    avg_sasa_patch_thr=0.75,
    cdrs=None,
):
    structure = _parse_structure(pdb)
    residues = []
    chains = [binder_chain] if binder_chain else [chain.id for chain in _model(structure)]
    for ch in chains:
        residues.extend(_chain_residues(structure, ch))
    residue_sasa = _residue_sasa(pdb, binder_chain if binder_chain else None)
    hydrophobic_aa = set(hydrophobic_aa or ["LEU", "ILE", "PHE", "TRP", "VAL", "MET", "TYR", "ALA"])

    scores = [
        (
            residue_sasa.get((res.parent.id, _residue_number(res)), 0.0) / 10.0
            if res.resname in hydrophobic_aa
            else 0.0
        )
        for res in residues
    ]
    hydrophobic_patches = []
    exposed_hydrophobic = []
    centers = [_residue_center(res) for res in residues]
    for idx, res in enumerate(residues):
        score = scores[idx]
        if res.resname not in hydrophobic_aa or score < limit_sasa:
            continue
        exposed_hydrophobic.append((idx + 1, res.resname))
        nearby = [
            (j + 1, residues[j].resname)
            for j, center in enumerate(centers)
            if np.linalg.norm(center - centers[idx]) <= patch_radius
        ]
        patch_score = float(np.mean([scores[j - 1] for j, _ in nearby])) if nearby else 0.0
        if patch_score >= avg_sasa_patch_thr:
            nearby_set = set(nearby)
            if not any(len(nearby_set & set(existing[1])) >= 2 for existing in hydrophobic_patches):
                hydrophobic_patches.append((patch_score, nearby))

    scores_array = np.array(scores)
    if cdrs is not None:
        valid = [i for i in cdrs if 0 <= i < len(scores_array)]
        cdr_sap = float(np.sum(scores_array[valid])) if valid else 0.0
    else:
        cdr_sap = float(np.sum(scores_array))
    return float(np.sum(scores_array)), cdr_sap, exposed_hydrophobic, hydrophobic_patches


def align_pdbs(reference_pdb, align_pdb, reference_chain_id, align_chain_id):
    reference_chain_id = reference_chain_id.split(",")[0]
    align_chain_id = align_chain_id.split(",")[0]
    ref_structure = _parse_structure(reference_pdb)
    mob_structure = _parse_structure(align_pdb)
    ref_res = _chain_residues(ref_structure, reference_chain_id)
    mob_res = _chain_residues(mob_structure, align_chain_id)
    ref_atoms, mob_atoms = [], []
    for r, m in zip(ref_res, mob_res):
        if "CA" in r and "CA" in m:
            ref_atoms.append(r["CA"])
            mob_atoms.append(m["CA"])
    if len(ref_atoms) < 3:
        raise ValueError("Not enough matched CA pairs to superimpose")
    sup = Superimposer()
    sup.set_atoms(ref_atoms, mob_atoms)
    sup.apply(mob_structure.get_atoms())
    io = PDBIO()
    io.set_structure(mob_structure)
    io.save(align_pdb)
    clean_pdb(align_pdb)


def unaligned_rmsd(reference_pdb, align_pdb, reference_chain_id, align_chain_id):
    reference_chain_id = reference_chain_id.split(",")[0]
    align_chain_id = align_chain_id.split(",")[0]
    ref_structure = _parse_structure(reference_pdb)
    mob_structure = _parse_structure(align_pdb)
    ref_res = _chain_residues(ref_structure, reference_chain_id)
    mob_res = _chain_residues(mob_structure, align_chain_id)
    diffs = []
    for r, m in zip(ref_res, mob_res):
        if "CA" in r and "CA" in m:
            diffs.append(np.sum((r["CA"].coord - m["CA"].coord) ** 2))
    if not diffs:
        return 100
    return round(float(np.sqrt(np.mean(diffs))), 2)


def get_residue_contacts(pdb_path, chain1="A", chain2="B", cutoff_distance=4.0):
    contacts = defaultdict(list)
    structure = _parse_structure(pdb_path)
    residues1 = _chain_residues(structure, chain1)
    residues2 = _chain_residues(structure, chain2)
    for res1 in residues1:
        for res2 in residues2:
            atoms1 = _heavy_atoms(res1)
            atoms2 = _heavy_atoms(res2)
            if not atoms1 or not atoms2:
                continue
            min_distance = min(_atom_distance(a, b) for a in atoms1 for b in atoms2)
            if min_distance <= cutoff_distance:
                types = {"VDW Contact"}
                if res1.resname in HYDROPHOBIC_3 and res2.resname in HYDROPHOBIC_3:
                    types.add("Hydrophobic")
                if any(
                    _atom_distance(a, b) <= 3.5
                    for a in atoms1
                    for b in atoms2
                    if a.element in DONOR_ACCEPTOR and b.element in DONOR_ACCEPTOR
                ):
                    types.add("H-bond")
                contacts[(_residue_number(res1), _residue_number(res2))] = {
                    "distance": min_distance,
                    "types": sorted(types),
                }
    return contacts


def find_nearby_residues_from_pdb(
    pdb_path: str,
    target_residues: Iterable[int],
    distance_threshold: float = 6.0,
    chain: str = "A",
):
    structure = _parse_structure(pdb_path)
    residues = _chain_residues(structure, chain)
    if isinstance(target_residues, int):
        target_residues = [target_residues]
    target_set = set(int(r) for r in target_residues)
    nearby = set()
    centers = {i + 1: _residue_center(res) for i, res in enumerate(residues)}
    for target in target_set:
        if target not in centers:
            continue
        for idx, center in centers.items():
            if idx == target or np.linalg.norm(center - centers[target]) <= distance_threshold:
                nearby.add(idx)
    return np.array(sorted(nearby))
