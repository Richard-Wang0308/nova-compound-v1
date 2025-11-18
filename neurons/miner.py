import os
import sys
import math
import random
import argparse
import asyncio
import datetime
import tempfile
import traceback
import base64
import hashlib
import sqlite3
import time

from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
import bittensor as bt
from boltz.wrapper import BoltzWrapper
from bittensor.core.errors import MetadataError
from substrateinterface import SubstrateInterface
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from config.config_loader import load_config
from utils import (
    get_sequence_from_protein_code,
    upload_file_to_github,
    get_challenge_params_from_blockhash,
    get_heavy_atom_count,
    # compute_maccs_entropy,  # Not needed with num_molecules=1 (entropy requires multiple molecules)
    get_smiles,
    molecule_unique_for_protein_hf,
    # find_chemically_identical,  # Not used in miner
)
from btdr import QuicknetBittensorDrandTimelock
from rdkit import Chem
from rdkit.Chem import Descriptors
from functools import lru_cache

# Global Boltz model instance (initialized lazily)
boltz = None

# ----------------------------------------------------------------------------
# 1. CONFIG & ARGUMENT PARSING
# ----------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """
    Parses command line arguments and merges with config defaults.

    Returns:
        argparse.Namespace: The combined configuration object.
    """
    parser = argparse.ArgumentParser()
    # Add override arguments for network.
    parser.add_argument('--network', default=os.getenv('SUBTENSOR_NETWORK'), help='Network to use')
    # Adds override arguments for netuid.
    parser.add_argument('--netuid', type=int, default=68, help="The chain subnet uid.")
    # Bittensor standard argument additions.
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)

    # Parse combined config
    config = bt.config(parser)

    # Load protein selection params
    config.update(load_config())

    # Final logging dir
    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey_str,
            config.netuid,
            'miner',
        )
    )

    # Ensure the logging directory exists.
    os.makedirs(config.full_path, exist_ok=True)
    return config


def load_github_path() -> str:
    """
    Constructs the path for GitHub operations from environment variables.
    
    Returns:
        str: The fully qualified GitHub path (owner/repo/branch/path).
    Raises:
        ValueError: If the final path exceeds 100 characters.
    """
    github_repo_name = os.environ.get('GITHUB_REPO_NAME')
    github_repo_branch = os.environ.get('GITHUB_REPO_BRANCH')
    github_repo_owner = os.environ.get('GITHUB_REPO_OWNER')
    github_repo_path = os.environ.get('GITHUB_REPO_PATH', '')

    if github_repo_name is None or github_repo_branch is None or github_repo_owner is None:
        raise ValueError("Missing one or more GitHub environment variables (GITHUB_REPO_*)")

    if github_repo_path == "":
        github_path = f"{github_repo_owner}/{github_repo_name}/{github_repo_branch}"
    else:
        github_path = f"{github_repo_owner}/{github_repo_name}/{github_repo_branch}/{github_repo_path}"

    if len(github_path) > 100:
        raise ValueError("GitHub path is too long. Please shorten it to 100 characters or less.")

    return github_path


# ----------------------------------------------------------------------------
# 2. LOGGING SETUP
# ----------------------------------------------------------------------------

def setup_logging(config: argparse.Namespace) -> None:
    """
    Sets up Bittensor logging.

    Args:
        config (argparse.Namespace): The miner configuration object.
    """
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(f"Running miner for subnet: {config.netuid} on network: {config.subtensor.network} with config:")
    bt.logging.info(config)


# ----------------------------------------------------------------------------
# 3. BITTENSOR & NETWORK SETUP
# ----------------------------------------------------------------------------

async def setup_bittensor_objects(config: argparse.Namespace) -> Tuple[Any, Any, Any, int, int]:
    """
    Initializes wallet, subtensor, and metagraph. Fetches the epoch length
    and calculates the miner UID.

    Args:
        config (argparse.Namespace): The miner configuration object.

    Returns:
        tuple: A 5-element tuple of
            (wallet, subtensor, metagraph, miner_uid, epoch_length).
    """
    bt.logging.info("Setting up Bittensor objects.")

    # Initialize wallet
    wallet = bt.wallet(config=config)
    bt.logging.info(f"Wallet: {wallet}")

    # Initialize subtensor (asynchronously)
    try:
        subtensor = bt.async_subtensor(network=config.network)
        await subtensor.initialize()
        bt.logging.info(f"Connected to subtensor network: {config.network}")
        
        # Sync metagraph
        metagraph = await subtensor.metagraph(config.netuid)
        await metagraph.sync()
        bt.logging.info(f"Metagraph synced successfully.")

        bt.logging.info(f"Subtensor: {subtensor}")
        bt.logging.info(f"Metagraph synced: {metagraph}")

        # Get miner UID - check if registered first
        hotkey_ss58 = wallet.hotkey.ss58_address
        if hotkey_ss58 not in metagraph.hotkeys:
            raise ValueError(f"Hotkey {hotkey_ss58} is not registered on netuid {config.netuid}. Please register and stake first.")
        miner_uid = metagraph.hotkeys.index(hotkey_ss58)
        bt.logging.info(f"Miner UID: {miner_uid}")

        # Query epoch length
        node = SubstrateInterface(url=config.network)
        # Set epoch_length to tempo + 1
        epoch_length = node.query("SubtensorModule", "Tempo", [config.netuid]).value + 1
        bt.logging.info(f"Epoch length query successful: {epoch_length} blocks")

        return wallet, subtensor, metagraph, miner_uid, epoch_length
    except Exception as e:
        bt.logging.error(f"Failed to setup Bittensor objects: {e}")
        bt.logging.error("Please check your network connection and the subtensor network status")
        raise

# ----------------------------------------------------------------------------
# 4. CACHING AND UTILITIES
# ----------------------------------------------------------------------------

# Cache for reaction molecule pools (static data, doesn't change)
_reaction_pools_cache: Dict[Tuple[str, int], Tuple[List[int], List[int], List[int]]] = {}

@lru_cache(maxsize=200_000)
def get_smiles_cached(name: str) -> Optional[str]:
    """Cache SMILES retrieval to avoid repeated database queries."""
    try:
        return get_smiles(name)
    except Exception:
        return None

@lru_cache(maxsize=100_000)
def mol_from_smiles_cached(s: str):
    """Cache molecule parsing to avoid repeated SMILES parsing."""
    if not s:
        return None
    try:
        return Chem.MolFromSmiles(s)
    except Exception:
        return None

@lru_cache(maxsize=200_000)
def smiles_to_inchikey_cached(s: str) -> Optional[str]:
    """Cache InChIKey generation."""
    if not s:
        return None
    try:
        mol = mol_from_smiles_cached(s)
        if mol is None:
            return None
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None

def _parse_components(name: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Parse reaction components from name format: 'rxn:{rxn_id}:{A}:{B}' or 'rxn:{rxn_id}:{A}:{B}:{C}'"""
    parts = name.split(":")
    if len(parts) < 4:
        return None, None, None
    try:
        A = int(parts[2])
        B = int(parts[3])
        C = int(parts[4]) if len(parts) > 4 else None
        return A, B, C
    except (ValueError, IndexError):
        return None, None, None

# ----------------------------------------------------------------------------
# 5. GENETIC ALGORITHM FUNCTIONS
# ----------------------------------------------------------------------------

def generate_offspring_from_elites(
    rxn_id: int,
    n: int,
    elite_names: List[str],
    molecules_A: List[int],
    molecules_B: List[int],
    molecules_C: List[int],
    mutation_prob: float = 0.1,
    seed: Optional[int] = None,
    avoid_names: Optional[set] = None,
    avoid_inchikeys: Optional[set] = None,
    max_tries: int = 10,
    neighborhood_limit: int = 5
) -> List[str]:
    """
    Generate offspring molecules from elite parents with mutation probability.
    Uses neighborhood expansion to explore around good solutions.
    
    Args:
        rxn_id: Reaction ID
        n: Number of offspring to generate
        elite_names: List of elite molecule names
        molecules_A, molecules_B, molecules_C: Molecule pools for each role
        mutation_prob: Probability of mutating away from elite (0-1)
        seed: Random seed for reproducibility
        avoid_names: Set of molecule names to avoid
        avoid_inchikeys: Set of InChIKeys to avoid
        max_tries: Maximum attempts to generate a valid offspring
        neighborhood_limit: Range to expand around each elite ID
    """
    rng = random.Random(seed) if seed is not None else random
    
    # Extract elite component IDs
    elite_As, elite_Bs, elite_Cs = set(), set(), set()
    for name in elite_names:
        A, B, C = _parse_components(name)
        if A is not None:
            elite_As.add(A)
        if B is not None:
            elite_Bs.add(B)
        if C is not None:
            elite_Cs.add(C)
    
    # Convert pools to sets for fast lookup
    pool_A_set = set(molecules_A)
    pool_B_set = set(molecules_B)
    pool_C_set = set(molecules_C)
    
    # Expand elite sets with neighborhoods
    def expand_with_neighborhood(elite_set: set, pool_set: set, limit: int) -> set:
        """Expand elite IDs to include neighboring IDs within the limit."""
        expanded = set(elite_set)
        for elite_id in elite_set:
            for neighbor_id in range(elite_id - limit, elite_id + limit + 1):
                if neighbor_id in pool_set:
                    expanded.add(neighbor_id)
        return expanded
    
    # Expand elite sets if neighborhood_limit > 0
    if neighborhood_limit > 0:
        if elite_As:
            elite_As = expand_with_neighborhood(elite_As, pool_A_set, neighborhood_limit)
        if elite_Bs:
            elite_Bs = expand_with_neighborhood(elite_Bs, pool_B_set, neighborhood_limit)
        if elite_Cs:
            elite_Cs = expand_with_neighborhood(elite_Cs, pool_C_set, neighborhood_limit)
    
    # Convert to lists for random.choice
    elite_As_list = list(elite_As) if elite_As else []
    elite_Bs_list = list(elite_Bs) if elite_Bs else []
    elite_Cs_list = list(elite_Cs) if elite_Cs else []
    
    out = []
    local_names = set()
    for _ in range(n):
        cand = None
        name = None
        for _try in range(max_tries):
            use_mutA = (not elite_As) or (not elite_As_list) or (rng.random() < mutation_prob)
            use_mutB = (not elite_Bs) or (not elite_Bs_list) or (rng.random() < mutation_prob)
            use_mutC = (not elite_Cs) or (not elite_Cs_list) or (rng.random() < mutation_prob)
            
            A = rng.choice(molecules_A) if use_mutA else rng.choice(elite_As_list)
            B = rng.choice(molecules_B) if use_mutB else rng.choice(elite_Bs_list)
            if molecules_C:
                C = rng.choice(molecules_C) if use_mutC else rng.choice(elite_Cs_list)
                name = f"rxn:{rxn_id}:{A}:{B}:{C}"
            else:
                name = f"rxn:{rxn_id}:{A}:{B}"
            
            if avoid_names and name in avoid_names:
                continue
            if name in local_names:
                continue
            
            if avoid_inchikeys:
                try:
                    s = get_smiles_cached(name)
                    if s:
                        key = smiles_to_inchikey_cached(s)
                        if key and key in avoid_inchikeys:
                            continue
                except Exception:
                    pass
            
            cand = name
            break
        
        if cand is None:
            # Fallback: generate a default name
            A = rng.choice(molecules_A)
            B = rng.choice(molecules_B)
            if molecules_C:
                C = rng.choice(molecules_C)
                name = f"rxn:{rxn_id}:{A}:{B}:{C}"
            else:
                name = f"rxn:{rxn_id}:{A}:{B}"
            cand = name
        
        out.append(cand)
        local_names.add(cand)
        if avoid_names is not None:
            avoid_names.add(cand)
    
    return out

# ----------------------------------------------------------------------------
# 6. MOLECULE GENERATION FROM MOL-RXN-DB (Enhanced with GA)
# ----------------------------------------------------------------------------

def get_reaction_molecule_pools(db_path: str, rxn_id: int) -> Tuple[List[int], List[int], List[int]]:
    """
    Get molecule pools for each role in a reaction.
    Uses caching since reaction pools are static and don't change.
    
    Returns:
        Tuple of (mols_A, mols_B, mols_C) lists
    """
    # Check cache first (reaction pools are static)
    cache_key = (db_path, rxn_id)
    if cache_key in _reaction_pools_cache:
        return _reaction_pools_cache[cache_key]
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get reaction info to understand role requirements
        cursor.execute("SELECT roleA, roleB, roleC FROM reactions WHERE rxn_id = ?", (rxn_id,))
        result = cursor.fetchone()
        if not result:
            bt.logging.error(f"Reaction {rxn_id} not found in database")
            conn.close()
            return [], [], []
        
        roleA, roleB, roleC = result
        
        # Get all molecules that can fulfill each role
        cursor.execute("SELECT mol_id FROM molecules WHERE (role_mask & ?) = ?", (roleA, roleA))
        mols_A = [row[0] for row in cursor.fetchall()]
        
        cursor.execute("SELECT mol_id FROM molecules WHERE (role_mask & ?) = ?", (roleB, roleB))
        mols_B = [row[0] for row in cursor.fetchall()]
        
        cursor.execute("SELECT mol_id FROM molecules WHERE (role_mask & ?) = ?", (roleC, roleC))
        mols_C = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        
        # Cache the result
        result_tuple = (mols_A, mols_B, mols_C)
        _reaction_pools_cache[cache_key] = result_tuple
        
        return result_tuple
        
    except Exception as e:
        bt.logging.error(f"Error getting reaction molecule pools: {e}")
        return [], [], []


def get_reaction_candidates(
    db_path: str, 
    rxn_id: int, 
    num_candidates: int = 1000,
    elite_names: Optional[List[str]] = None,
    elite_frac: float = 0.0,
    mutation_prob: float = 0.1,
    avoid_inchikeys: Optional[set] = None,
    neighborhood_limit: int = 0
) -> List[str]:
    """
    Generate reaction candidates from Mol-Rxn-DB, optionally using genetic algorithm.
    
    Args:
        db_path: Path to molecules.sqlite database
        rxn_id: Reaction ID (4 or 5)
        num_candidates: Number of combinations to generate
        elite_names: List of elite molecule names for GA
        elite_frac: Fraction of candidates to generate from elites (0-1)
        mutation_prob: Mutation probability for GA (0-1)
        avoid_inchikeys: Set of InChIKeys to avoid
        neighborhood_limit: Neighborhood expansion limit for elites
        
    Returns:
        List of reaction strings: ["rxn:4:123:456:789", ...]
    """
    try:
        mols_A, mols_B, mols_C = get_reaction_molecule_pools(db_path, rxn_id)
        
        if not (mols_A and mols_B and mols_C):
            bt.logging.warning(f"Not enough molecules for reaction {rxn_id}. A:{len(mols_A)}, B:{len(mols_B)}, C:{len(mols_C)}")
            return []
        
        candidates = []
        
        # Use GA if elites are provided
        if elite_names and elite_frac > 0:
            n_elite = max(0, min(num_candidates, int(num_candidates * elite_frac)))
            n_rand = num_candidates - n_elite
            
            # Generate elite-based offspring
            if n_elite > 0:
                elite_candidates = generate_offspring_from_elites(
                    rxn_id=rxn_id,
                    n=n_elite,
                    elite_names=elite_names,
                    molecules_A=mols_A,
                    molecules_B=mols_B,
                    molecules_C=mols_C,
                    mutation_prob=mutation_prob,
                    avoid_inchikeys=avoid_inchikeys,
                    neighborhood_limit=neighborhood_limit
                )
                candidates.extend(elite_candidates)
            
            # Generate random candidates
            if n_rand > 0:
                rng = random.Random()
                for _ in range(n_rand):
                    mol1 = rng.choice(mols_A)
                    mol2 = rng.choice(mols_B)
                    mol3 = rng.choice(mols_C)
                    candidates.append(f"rxn:{rxn_id}:{mol1}:{mol2}:{mol3}")
        else:
            # Pure random generation
            rng = random.Random()
            for _ in range(num_candidates):
                mol1 = rng.choice(mols_A)
                mol2 = rng.choice(mols_B)
                mol3 = rng.choice(mols_C)
                candidates.append(f"rxn:{rxn_id}:{mol1}:{mol2}:{mol3}")
        
        bt.logging.info(f"Generated {len(candidates)} candidates for reaction {rxn_id} (elite_frac={elite_frac:.2f})")
        return candidates
        
    except Exception as e:
        bt.logging.error(f"Error generating reaction candidates: {e}")
        return []


def validate_molecule(smiles: str, config: argparse.Namespace, weekly_target: str) -> bool:
    """
    Validate a molecule meets all requirements (using cached functions).
    
    Args:
        smiles: SMILES string
        config: Configuration object
        weekly_target: Target protein code
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # Check heavy atoms
        if get_heavy_atom_count(smiles) < config.min_heavy_atoms:
            return False
        
        # Check rotatable bonds (using cached mol)
        mol = mol_from_smiles_cached(smiles)
        if mol is None:
            return False
        
        num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        if num_rotatable_bonds < config.min_rotatable_bonds or num_rotatable_bonds > config.max_rotatable_bonds:
            return False
        
        # Check uniqueness for protein
        if not molecule_unique_for_protein_hf(weekly_target, smiles):
            return False
        
        return True
        
    except Exception as e:
        bt.logging.debug(f"Validation error for {smiles}: {e}")
        return False


# ----------------------------------------------------------------------------
# 5. INFERENCE AND SUBMISSION LOGIC
# ----------------------------------------------------------------------------

async def run_boltz_model_loop(state: Dict[str, Any], final_block_hash: str, config: argparse.Namespace) -> None:
    """
    Continuously runs the Boltz model on batches of molecules generated from Mol-Rxn-DB.
    Updates the best candidate whenever a higher score is found, but only submits when close to epoch end.

    Args:
        state (dict): A shared state dict containing references to:
            'config', 'current_challenge_targets', 'best_score',
            'candidate_product', 'subtensor', 'epoch_length',
            'last_submitted_product', 'shutdown_event', etc.
        final_block_hash: Block hash for deterministic scoring
        config: Configuration object
    """
    global boltz
    
    bt.logging.info("Starting Boltz model inference loop.")
    
    # Initialize Boltz model
    if boltz is None:
        bt.logging.info("Initializing Boltz model...")
        boltz = BoltzWrapper()
        bt.logging.info("Boltz model initialized successfully")
    
    # Get target protein
    weekly_target = config.weekly_target
    if not state.get('current_challenge_targets'):
        bt.logging.warning("No target proteins available, waiting...")
        return
    
    target_protein = state['current_challenge_targets'][0]
    bt.logging.info(f"Scoring molecules for target protein: {target_protein}")
    
    # Database path
    db_path = os.path.join(BASE_DIR, "combinatorial_db", "molecules.sqlite")
    if not os.path.exists(db_path):
        bt.logging.error(f"Database not found at {db_path}")
        return
    
    # Process in batches - OPTIMIZED FOR RTX 4090 HIGH PERFORMANCE
    batch_size = 128  # Large batch size for RTX 4090 GPU utilization
    candidates_per_batch = 2000  # Generate 2000 candidates at a time (1000 per reaction) - MAXIMIZED for RTX 4090
    
    # Elite pool management - larger for better diversity
    elite_pool_size = 500  # Keep top 500 candidates - MAXIMIZED for RTX 4090
    elite_pool = pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
    seen_inchikeys = set()
    
    # Adaptive GA parameters - optimized for exploration
    mutation_prob = 0.3  # Start with higher mutation for more exploration
    elite_frac = 0.15  # Lower elite fraction to explore more initially
    neighborhood_limit = 5  # Start with neighborhood expansion enabled
    iteration = 0
    prev_mean_score = None
    start_time = time.time()
    
    while not state['shutdown_event'].is_set():
        iteration += 1
        try:
            # Get elite names for GA
            elite_names = elite_pool["name"].tolist() if not elite_pool.empty else None
            
            # Generate new candidates on-demand (with GA support)
            candidates_rxn4 = get_reaction_candidates(
                db_path, rxn_id=4, num_candidates=candidates_per_batch // 2,
                elite_names=elite_names, elite_frac=elite_frac,
                mutation_prob=mutation_prob, avoid_inchikeys=seen_inchikeys,
                neighborhood_limit=neighborhood_limit
            )
            candidates_rxn5 = get_reaction_candidates(
                db_path, rxn_id=5, num_candidates=candidates_per_batch // 2,
                elite_names=elite_names, elite_frac=elite_frac,
                mutation_prob=mutation_prob, avoid_inchikeys=seen_inchikeys,
                neighborhood_limit=neighborhood_limit
            )
            all_candidates = candidates_rxn4 + candidates_rxn5
            
            if not all_candidates:
                bt.logging.warning("No candidates generated, waiting...")
                await asyncio.sleep(0.1)  # Minimal wait for RTX 4090
                continue
            
            # Process in batches - accumulate for larger GPU batches
            processed = 0
            accumulated_valid = []  # Accumulate valid molecules for larger scoring batches
            accumulated_names = []
            accumulated_inchikeys = []
            accumulation_target = batch_size * 4  # Accumulate 4x batch_size before scoring (RTX 4090 can handle it)
            
            while not state['shutdown_event'].is_set() and processed < len(all_candidates):
                # Get next batch
                batch_end = min(processed + batch_size, len(all_candidates))
                batch = all_candidates[processed:batch_end]
                processed = batch_end
                
                # Validate and convert to SMILES (using cached functions)
                valid_molecules = []
                valid_names = []
                valid_inchikeys = []
                
                for candidate in batch:
                    try:
                        smiles = get_smiles_cached(candidate)  # Use cached function
                        if not smiles:
                            continue
                        
                        # Check InChIKey first to avoid duplicates (faster than validation)
                        inchikey = smiles_to_inchikey_cached(smiles)
                        if inchikey and inchikey in seen_inchikeys:
                            continue
                        
                        if validate_molecule(smiles, config, weekly_target):
                            valid_molecules.append(smiles)
                            valid_names.append(candidate)
                            valid_inchikeys.append(inchikey)
                    except Exception as e:
                        bt.logging.debug(f"Error processing candidate {candidate}: {e}")
                        continue
                
                if not valid_molecules:
                    continue
                
                # Additional deduplication check (InChIKey-based, more efficient)
                unique_data = {}
                for name, smiles, inchikey in zip(valid_names, valid_molecules, valid_inchikeys):
                    if inchikey and inchikey not in unique_data:
                        unique_data[inchikey] = (name, smiles)
                
                if not unique_data:
                    continue
                
                valid_names = [v[0] for v in unique_data.values()]
                valid_molecules = [v[1] for v in unique_data.values()]
                valid_inchikeys = list(unique_data.keys())
                
                # Accumulate for larger GPU batches (RTX 4090 optimization)
                accumulated_valid.extend(valid_molecules)
                accumulated_names.extend(valid_names)
                accumulated_inchikeys.extend(valid_inchikeys)
                
                # Score when we have enough accumulated or at end of candidates
                should_score = (len(accumulated_valid) >= accumulation_target or 
                              processed >= len(all_candidates))
                
                if not should_score:
                    continue
                
                # Use accumulated batch for scoring
                valid_molecules = accumulated_valid
                valid_names = accumulated_names
                valid_inchikeys = accumulated_inchikeys
                
                # Reset accumulation
                accumulated_valid = []
                accumulated_names = []
                accumulated_inchikeys = []
                
                # Score with Boltz (larger batch for RTX 4090)
                valid_molecules_by_uid = {
                    0: {"smiles": valid_molecules, "names": valid_names}
                }
                
                score_dict = {
                    0: {
                        # "entropy": None,  # Not needed for miner (only used by validator)
                        "entropy_boltz": None,  # Required by boltz wrapper (will be set by wrapper)
                        "block_submitted": None,
                        "push_time": ""
                    }
                }
                
                try:
                    boltz.score_molecules_target(
                        valid_molecules_by_uid,
                        score_dict,
                        config.__dict__,
                        final_block_hash
                    )
                    
                    boltz_score = score_dict[0].get('boltz_score')
                    
                    if boltz_score is not None and math.isfinite(boltz_score):
                        # Get per-molecule scores if available
                        per_molecule_metric = getattr(boltz, 'per_molecule_metric', {})
                        batch_scores = []
                        
                        if per_molecule_metric and 0 in per_molecule_metric:
                            # Get individual scores for each molecule
                            for idx, smiles in enumerate(valid_molecules):
                                mol_score = per_molecule_metric[0].get(smiles)
                                if mol_score is not None and math.isfinite(mol_score):
                                    batch_scores.append({
                                        "name": valid_names[idx],
                                        "smiles": smiles,
                                        "InChIKey": valid_inchikeys[idx] if idx < len(valid_inchikeys) else None,
                                        "score": mol_score
                                    })
                        else:
                            # Use average score for all molecules
                            for idx, smiles in enumerate(valid_molecules):
                                batch_scores.append({
                                    "name": valid_names[idx],
                                    "smiles": smiles,
                                    "InChIKey": valid_inchikeys[idx] if idx < len(valid_inchikeys) else None,
                                    "score": boltz_score
                                })
                        
                        if batch_scores:
                            # Update seen InChIKeys
                            for item in batch_scores:
                                if item["InChIKey"]:
                                    seen_inchikeys.add(item["InChIKey"])
                            
                            # Merge with elite pool, deduplicate, sort, and take top N
                            batch_df = pd.DataFrame(batch_scores)
                            elite_pool = pd.concat([elite_pool, batch_df])
                            elite_pool = elite_pool.drop_duplicates(subset=["InChIKey"], keep="first")
                            elite_pool = elite_pool.sort_values(by="score", ascending=False)
                            elite_pool = elite_pool.head(elite_pool_size)
                            
                            # Update best candidate (top of elite pool)
                            if not elite_pool.empty:
                                best_row = elite_pool.iloc[0]
                                best_score = best_row["score"]
                                best_candidate = best_row["name"]
                                
                                if best_score > state['best_score']:
                                    state['best_score'] = best_score
                                    state['candidate_product'] = best_candidate
                                    bt.logging.info(f"New best score: {best_score:.4f}, Candidate: {best_candidate}, Elite pool size: {len(elite_pool)}")
                            
                            # Adaptive GA parameters - optimized for competition
                            if iteration > 2 and not elite_pool.empty:  # Start adapting earlier
                                current_mean = elite_pool["score"].mean()
                                
                                # Adjust based on duplicate ratio - prioritize exploration in competition
                                dup_ratio = (len(batch_scores) - len(batch_df)) / max(1, len(batch_scores))
                                if dup_ratio > 0.5:  # Lower threshold - more aggressive exploration
                                    mutation_prob = min(0.6, mutation_prob * 1.4)  # Allow higher mutation
                                    elite_frac = max(0.1, elite_frac * 0.75)  # Lower elite fraction
                                    neighborhood_limit = min(10, neighborhood_limit + 1)  # Expand neighborhood
                                elif dup_ratio < 0.15 and not elite_pool.empty:
                                    mutation_prob = max(0.1, mutation_prob * 0.95)  # Slightly reduce mutation
                                    elite_frac = min(0.4, elite_frac * 1.05)  # Slightly increase elite
                                
                                # Adjust based on score improvement - more exploration-focused
                                if prev_mean_score is not None:
                                    score_improvement = max(0, (current_mean - prev_mean_score) / max(abs(prev_mean_score), 1e-6))
                                    if score_improvement < 0.03:  # Lower threshold - explore more aggressively
                                        # Explore more - competition needs diversity
                                        mutation_prob = min(0.6, mutation_prob * 1.4)
                                        elite_frac = max(0.1, elite_frac * 0.8)
                                        neighborhood_limit = min(10, neighborhood_limit + 1)
                                    elif score_improvement > 0.25:  # Only exploit if significant improvement
                                        # Exploit more - but still maintain some exploration
                                        mutation_prob = max(0.15, mutation_prob * 0.95)
                                        elite_frac = min(0.4, elite_frac * 1.05)
                                
                                prev_mean_score = current_mean
                            
                            # Log exploration stats periodically (more frequent for RTX 4090)
                            if iteration % 5 == 0:  # More frequent logging
                                elapsed = time.time() - start_time
                                molecules_explored = len(seen_inchikeys)
                                molecules_per_sec = molecules_explored / max(elapsed, 1)
                                bt.logging.info(
                                    f"[RTX 4090] Exploration: {molecules_explored} unique molecules, "
                                    f"{iteration} iterations, {elapsed:.1f}s elapsed, "
                                    f"{molecules_per_sec:.1f} mol/s, "
                                    f"mutation={mutation_prob:.2f}, elite_frac={elite_frac:.2f}, "
                                    f"neighborhood={neighborhood_limit}, pool_size={len(elite_pool)}"
                                )
                    
                except Exception as e:
                    bt.logging.error(f"Error scoring with Boltz: {e}")
                    traceback.print_exc()
                    continue
                
                # Check if time to submit
                try:
                    current_block = await state['subtensor'].get_current_block()
                    next_epoch_block = ((current_block // state['epoch_length']) + 1) * state['epoch_length']
                    blocks_until_epoch = next_epoch_block - current_block
                    
                    if state['candidate_product'] and blocks_until_epoch <= 20:
                        bt.logging.info(f"Close to epoch end ({blocks_until_epoch} blocks remaining), attempting submission...")
                        if state['candidate_product'] != state.get('last_submitted_product'):
                            bt.logging.info("Attempting to submit new candidate...")
                            try:
                                await submit_response(state)
                            except Exception as e:
                                bt.logging.error(f"Error submitting response: {e}")
                        else:
                            bt.logging.info("Skipping submission - same product as last submission")
                except Exception as e:
                    bt.logging.debug(f"Error checking submission timing: {e}")
                
                # Minimal sleep for RTX 4090 - maximize GPU utilization
                try:
                    current_block = await state['subtensor'].get_current_block()
                    next_epoch_block = ((current_block // state['epoch_length']) + 1) * state['epoch_length']
                    blocks_until_epoch = next_epoch_block - current_block
                    
                    # Only minimal sleep if we have plenty of time (RTX 4090 can handle continuous load)
                    if blocks_until_epoch > 100:
                        await asyncio.sleep(0.1)  # Minimal sleep for RTX 4090
                    # No sleep if close to epoch end - maximize exploration
                except Exception:
                    await asyncio.sleep(0.1)  # Minimal fallback sleep
            
        except Exception as e:
            bt.logging.error(f"Error in Boltz model loop: {e}")
            traceback.print_exc()
            await asyncio.sleep(0.1)  # Minimal error recovery for RTX 4090
    
    bt.logging.info("Boltz model loop completed.")


async def submit_response(state: Dict[str, Any]) -> None:
    """
    Encrypts and submits the current candidate product as a chain commitment and uploads
    the encrypted response to GitHub. If the chain accepts the commitment, we finalize it.

    Args:
        state (dict): Shared state dictionary containing references to:
            'bdt', 'miner_uid', 'candidate_product', 'subtensor', 'wallet', 'config',
            'github_path', etc.
    """
    candidate_product = state['candidate_product']
    if not candidate_product:
        bt.logging.warning("No candidate product to submit")
        return

    bt.logging.info(f"Starting submission process for product: {candidate_product}")
    
    # 1) Encrypt the response
    current_block = await state['subtensor'].get_current_block()
    encrypted_response = state['bdt'].encrypt(state['miner_uid'], candidate_product, current_block)
    bt.logging.info(f"Encrypted response generated successfully")

    # 2) Create temp file, write content
    tmp_file = tempfile.NamedTemporaryFile(delete=True)
    with open(tmp_file.name, 'w+') as f:
        f.write(str(encrypted_response))
        f.flush()

        # Read, base64-encode
        f.seek(0)
        content_str = f.read()
        encoded_content = base64.b64encode(content_str.encode()).decode()

        # Generate short hash-based filename
        filename = hashlib.sha256(content_str.encode()).hexdigest()[:20]
        commit_content = f"{state['github_path']}/{filename}.txt"
        bt.logging.info(f"Prepared commit content: {commit_content}")

        # 3) Attempt chain commitment
        bt.logging.info(f"Attempting chain commitment...")
        try: 
            commitment_status = await state['subtensor'].set_commitment(
                wallet=state['wallet'],
                netuid=state['config'].netuid,
                data=commit_content
            )
            bt.logging.info(f"Chain commitment status: {commitment_status}")
        except MetadataError:
            bt.logging.info("Too soon to commit again. Will keep looking for better candidates.")
            return

        # 4) If chain commitment success, upload to GitHub
        if commitment_status:
            try:
                bt.logging.info(f"Commitment set successfully for {commit_content}")
                bt.logging.info("Attempting GitHub upload...")
                github_status = upload_file_to_github(filename, encoded_content)
                if github_status:
                    bt.logging.info(f"File uploaded successfully to {commit_content}")
                    state['last_submitted_product'] = candidate_product
                    state['last_submission_time'] = datetime.datetime.now()
                else:
                    bt.logging.error(f"Failed to upload file to GitHub for {commit_content}")
            except Exception as e:
                bt.logging.error(f"Failed to upload file for {commit_content}: {e}")


# ----------------------------------------------------------------------------
# 6. MAIN MINING LOOP
# ----------------------------------------------------------------------------

async def run_miner(config: argparse.Namespace) -> None:
    """
    The main mining loop, orchestrating:
      - Bittensor objects initialization
      - Model initialization
      - Fetching new proteins each epoch
      - Running inference and submissions
      - Periodically syncing metagraph

    Args:
        config (argparse.Namespace): The miner configuration object.
    """

    # 1) Setup wallet, subtensor, metagraph, etc.
    wallet, subtensor, metagraph, miner_uid, epoch_length = await setup_bittensor_objects(config)

    # 2) Prepare shared state
    state: Dict[str, Any] = {
        # environment / config
        'config': config,
        
        # GitHub
        'github_path': load_github_path(),

        # Bittensor
        'wallet': wallet,
        'subtensor': subtensor,
        'metagraph': metagraph,
        'miner_uid': miner_uid,
        'epoch_length': epoch_length,

        # Encryption
        'bdt': QuicknetBittensorDrandTimelock(),

        # Inference state
        'candidate_product': None,
        'best_score': float('-inf'),
        'last_submitted_product': None,
        'last_submission_time': None,
        'shutdown_event': asyncio.Event(),

        # Challenges
        'current_challenge_targets': [],
        'last_challenge_targets': [],
    }

    bt.logging.info("Entering main miner loop...")

    # 3) If we start mid-epoch, obtain most recent proteins from block hash
    current_block = await subtensor.get_current_block()
    last_boundary = (current_block // epoch_length) * epoch_length
    final_block_hash = await subtensor.determine_block_hash(current_block)
    next_boundary = last_boundary + epoch_length

    # If we start too close to epoch end, wait for next epoch
    if next_boundary - current_block < 20:
        bt.logging.info(f"Too close to epoch end, waiting for next epoch to start...")
        block_to_check = next_boundary
        await asyncio.sleep(12*10)
    else:
        block_to_check = last_boundary

    block_hash = await subtensor.determine_block_hash(block_to_check)
    startup_proteins = get_challenge_params_from_blockhash(
        block_hash=block_hash,
        weekly_target=config.weekly_target,
        num_antitargets=config.num_antitargets
    )

    if startup_proteins:
        state['current_challenge_targets'] = startup_proteins["targets"]
        state['last_challenge_targets'] = startup_proteins["targets"]
        bt.logging.info(f"Startup targets: {startup_proteins['targets']}")

        # 4) Launch the inference loop
        try:
            state['inference_task'] = asyncio.create_task(run_boltz_model_loop(state, final_block_hash, config))
            bt.logging.debug("Inference started on startup proteins.")
        except Exception as e:
            bt.logging.error(f"Error starting inference: {e}")

    # 5) Main epoch-based loop
    while True:
        try:
            current_block = await subtensor.get_current_block()

            # If we are at an epoch boundary, fetch new proteins
            if current_block % epoch_length == 0:
                bt.logging.info(f"Found epoch boundary at block {current_block}.")
                
                block_hash = await subtensor.determine_block_hash(current_block)
                final_block_hash = await subtensor.determine_block_hash(current_block)
                
                new_proteins = get_challenge_params_from_blockhash(
                    block_hash=block_hash,
                    weekly_target=config.weekly_target,
                    num_antitargets=config.num_antitargets
                )
                if (new_proteins and 
                    (new_proteins["targets"] != state['last_challenge_targets'])):
                    state['current_challenge_targets'] = new_proteins["targets"]
                    state['last_challenge_targets'] = new_proteins["targets"]
                    bt.logging.info(f"New proteins - targets: {new_proteins['targets']}")

                # Cancel old inference, reset relevant state
                if 'inference_task' in state and state['inference_task']:
                    if not state['inference_task'].done():
                        state['shutdown_event'].set()
                        bt.logging.debug("Shutdown event set for old inference task.")
                        await state['inference_task']

                # Reset best score and candidate
                state['candidate_product'] = None
                state['best_score'] = float('-inf')
                state['last_submitted_product'] = None
                state['shutdown_event'] = asyncio.Event()

                # Start new inference
                try:
                    state['inference_task'] = asyncio.create_task(run_boltz_model_loop(state, final_block_hash, config))
                    bt.logging.debug("New inference task started.")
                except Exception as e:
                    bt.logging.error(f"Error starting new inference: {e}")

            # Periodically update our knowledge of the network
            if current_block % 60 == 0:
                await metagraph.sync()
                log = (
                    f"Block: {metagraph.block.item()} | "
                    f"Number of nodes: {metagraph.n} | "
                    f"Current epoch: {metagraph.block.item() // epoch_length}"
                )
                bt.logging.info(log)

            await asyncio.sleep(1)

        except RuntimeError as e:
            bt.logging.error(e)
            traceback.print_exc()

        except KeyboardInterrupt:
            bt.logging.success("Keyboard interrupt detected. Exiting miner.")
            break


# ----------------------------------------------------------------------------
# 7. ENTRY POINT
# ----------------------------------------------------------------------------

async def main() -> None:
    """
    Main entry point for asynchronous execution of the miner logic.
    """
    config = parse_arguments()
    setup_logging(config)
    await run_miner(config)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
