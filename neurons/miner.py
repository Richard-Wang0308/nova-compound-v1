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
from collections import OrderedDict

# Canonical SMILES helper (cached) - reduced size to manage memory
@lru_cache(maxsize=50_000)
def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Canonicalize SMILES string for consistent key matching."""
    mol = mol_from_smiles_cached(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None

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
    # Performance tuning arguments
    parser.add_argument('--batch-size', type=int, default=None, help="Batch size for GPU scoring (default: 128 for RTX 4090)")
    parser.add_argument('--sleep-time', type=float, default=None, help="Sleep time between iterations in seconds (default: 0.1)")
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

@lru_cache(maxsize=20_000)
def get_smiles_cached(name: str) -> Optional[str]:
    """Cache SMILES retrieval to avoid repeated database queries."""
    try:
        return get_smiles(name)
    except Exception:
        return None

@lru_cache(maxsize=20_000)
def mol_from_smiles_cached(s: str) -> Optional[Chem.Mol]:
    """
    Cache molecule parsing to avoid repeated SMILES parsing (reduced size to manage memory).
    
    Note: lru_cache is thread-safe in Python 3.9+ when used with run_in_executor.
    For Python <3.9, consider using a thread-safe wrapper if issues arise.
    """
    if not s:
        return None
    try:
        return Chem.MolFromSmiles(s)
    except Exception:
        return None

@lru_cache(maxsize=50_000)
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
    """
    Parse reaction components from name format: 'rxn:{rxn_id}:{A}:{B}' or 'rxn:{rxn_id}:{A}:{B}:{C}'
    
    Args:
        name: Reaction name string
        
    Returns:
        Tuple of (A, B, C) component IDs, or (None, None, None) if parsing fails
    """
    if not name or not isinstance(name, str):
        return None, None, None
    
    parts = name.split(":")
    if len(parts) < 4:
        return None, None, None
    
    try:
        # Validate rxn_id format
        if parts[0] != "rxn":
            return None, None, None
        
        A = int(parts[2])
        B = int(parts[3])
        C = int(parts[4]) if len(parts) > 4 and parts[4] else None
        
        # Defensive check: ensure IDs are non-negative
        if A < 0 or B < 0 or (C is not None and C < 0):
            return None, None, None
        
        return A, B, C
    except (ValueError, IndexError, TypeError):
        return None, None, None

# ----------------------------------------------------------------------------
# 5. GENETIC ALGORITHM FUNCTIONS
# ----------------------------------------------------------------------------

def generate_offspring_from_elites(
    rxn_id: int,
    n: int,
    elite_names: List[str],
    elite_scores: Optional[Dict[str, float]],
    molecules_A: List[int],
    molecules_B: List[int],
    molecules_C: List[int],
    crossover_prob: float = 0.7,
    mutation_prob: float = 0.1,
    seed: Optional[int] = None,
    avoid_names: Optional[set] = None,
    avoid_inchikeys: Optional[set] = None,
    max_tries: int = 10,
    neighborhood_limit: int = 5
) -> List[str]:
    """
    Generate offspring molecules using proper GA operators: selection, crossover, and mutation.
    Based on canonical genetic algorithm principles (Whitley, 1993).
    
    GA Pipeline:
    1. Selection: Fitness-proportional selection from elite pool (if scores provided)
    2. Crossover: Uniform crossover between two parents (with probability crossover_prob)
    3. Mutation: Random component replacement (with probability mutation_prob per component)
    4. Neighborhood Expansion: Local exploration around elite component IDs (if neighborhood_limit > 0)
    5. Validation: Check against avoid lists and generate valid reaction strings
    
    Args:
        rxn_id: Reaction ID (4 or 5)
        n: Number of offspring to generate
        elite_names: List of elite molecule names (chromosomes) in format "rxn:{id}:{A}:{B}:{C}"
        elite_scores: Optional dict mapping elite names to scores (for fitness-proportional selection)
        molecules_A, molecules_B, molecules_C: Molecule pools for each role
        crossover_prob: Probability of crossover between parents (0-1)
        mutation_prob: Probability of mutating a component (0-1)
        seed: Random seed for reproducibility
        avoid_names: Set of molecule names to avoid
        avoid_inchikeys: Set of InChIKeys to avoid
        max_tries: Maximum attempts to generate a valid offspring
        neighborhood_limit: Range to expand around each elite ID (0 = no expansion)
    
    Returns:
        List of reaction strings: ["rxn:{id}:{A}:{B}:{C}", ...]
    """
    rng = random.Random(seed) if seed is not None else random
    
    if not elite_names:
        return []
    
    # Parse elite chromosomes (preserve parent structure) - build aligned lists
    elite_valid_names = []
    elite_chromosomes = []
    for name in elite_names:
        A, B, C = _parse_components(name)
        if A is not None and B is not None:
            elite_valid_names.append(name)  # Keep track of valid names
            elite_chromosomes.append((A, B, C))
    
    if not elite_chromosomes:
        return []
    
    # Convert pools to sets for fast lookup
    pool_A_set = set(molecules_A)
    pool_B_set = set(molecules_B)
    pool_C_set = set(molecules_C)
    
    # Fitness-proportional selection weights (if scores provided) - aligned with elite_chromosomes
    selection_weights = None
    if elite_scores:
        # Use elite_valid_names (aligned with elite_chromosomes) for score lookup
        scores_list = [elite_scores.get(name, 0.0) for name in elite_valid_names]
        min_score = min(scores_list) if scores_list else 0.0
        normalized = [s - min_score + 1e-6 for s in scores_list]  # Shift to positive
        total = sum(normalized)
        if total > 0:
            selection_weights = [w / total for w in normalized]
    
    def expand_with_neighborhood(component_id: int, pool_set: set, limit: int) -> List[int]:
        """
        Expand a single component ID to include neighbors (local, not permanent).
        
        Args:
            component_id: Base component ID
            pool_set: Set of valid component IDs in the pool
            limit: Maximum distance to expand (must be non-negative)
            
        Returns:
            List of component IDs including base and neighbors
        """
        if component_id is None or component_id < 0:
            return []
        if not pool_set or limit < 0:
            return [component_id] if component_id in pool_set else []
        
        expanded = [component_id] if component_id in pool_set else []
        # Defensive: ensure limit doesn't cause excessive iteration
        safe_limit = min(limit, 100)  # Cap at 100 to prevent excessive expansion
        for neighbor_id in range(component_id - safe_limit, component_id + safe_limit + 1):
            if neighbor_id != component_id and neighbor_id >= 0 and neighbor_id in pool_set:
                expanded.append(neighbor_id)
        return expanded if expanded else []
    
    def select_parent(weights: Optional[List[float]] = None) -> Tuple[int, int, int]:
        """Select a parent chromosome using fitness-proportional selection or uniform selection."""
        if weights and len(weights) == len(elite_chromosomes):
            idx = rng.choices(range(len(elite_chromosomes)), weights=weights, k=1)[0]
        else:
            idx = rng.randint(0, len(elite_chromosomes) - 1)
        return elite_chromosomes[idx]
    
    def crossover(parent1: Tuple[int, int, int], parent2: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Uniform crossover: randomly choose each component from parent1 or parent2."""
        A1, B1, C1 = parent1
        A2, B2, C2 = parent2
        # Uniform crossover at component level
        A = A1 if rng.random() < 0.5 else A2
        B = B1 if rng.random() < 0.5 else B2
        # Explicit C handling: prefer random choice if both available, otherwise use available one
        if C1 is not None and C2 is not None:
            C = C1 if rng.random() < 0.5 else C2
        elif C1 is not None:
            C = C1
        elif C2 is not None:
            C = C2
        else:
            C = None  # Explicit None when neither parent has C component
        return (A, B, C)
    
    def mutate(chromosome: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Mutate components with mutation_prob probability."""
        A, B, C = chromosome
        # Mutate each component independently
        if rng.random() < mutation_prob:
            A = rng.choice(molecules_A)
        if rng.random() < mutation_prob:
            B = rng.choice(molecules_B)
        if C is not None and rng.random() < mutation_prob:
            C = rng.choice(molecules_C)
        return (A, B, C)
    
    out = []
    local_names = set()
    for _ in range(n):
        cand = None
        name = None
        for _try in range(max_tries):
            # GA: Selection -> Crossover -> Mutation
            parent1 = select_parent(selection_weights)
            parent2 = select_parent(selection_weights) if len(elite_chromosomes) > 1 else parent1
            
            # Crossover with probability crossover_prob
            if rng.random() < crossover_prob and len(elite_chromosomes) > 1:
                offspring = crossover(parent1, parent2)
            else:
                offspring = parent1  # No crossover, use parent1
            
            # Apply mutation
            offspring = mutate(offspring)
            
            # Apply neighborhood expansion to offspring components (local exploration)
            A, B, C = offspring
            if neighborhood_limit > 0:
                if A in pool_A_set:
                    A_candidates = expand_with_neighborhood(A, pool_A_set, neighborhood_limit)
                    if A_candidates:  # Check for empty list to prevent IndexError
                        A = rng.choice(A_candidates)
                if B in pool_B_set:
                    B_candidates = expand_with_neighborhood(B, pool_B_set, neighborhood_limit)
                    if B_candidates:  # Check for empty list to prevent IndexError
                        B = rng.choice(B_candidates)
                if C is not None and C in pool_C_set:
                    C_candidates = expand_with_neighborhood(C, pool_C_set, neighborhood_limit)
                    if C_candidates:  # Check for empty list to prevent IndexError
                        C = rng.choice(C_candidates)
            
            # Construct chromosome string
            if C is not None:
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
            # Fallback: generate a random chromosome (with avoid check)
            for fallback_try in range(5):  # Limit fallback attempts
                A = rng.choice(molecules_A)
                B = rng.choice(molecules_B)
                if molecules_C:
                    C = rng.choice(molecules_C)
                    name = f"rxn:{rxn_id}:{A}:{B}:{C}"
                else:
                    name = f"rxn:{rxn_id}:{A}:{B}"
                
                # Check avoid lists in fallback
                if avoid_inchikeys:
                    try:
                        s = get_smiles_cached(name)
                        if s:
                            key = smiles_to_inchikey_cached(s)
                            if key and key in avoid_inchikeys:
                                continue  # Try again
                    except Exception:
                        pass
                
                # Check avoid_names
                if avoid_names and name in avoid_names:
                    continue
                
                cand = name
                break
            
            # If still None after fallback attempts, use last generated name
            if cand is None:
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
        # Thread-safe SQLite connection (for use with run_in_executor)
        conn = sqlite3.connect(db_path, check_same_thread=False)
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
    elite_scores: Optional[Dict[str, float]] = None,
    elite_frac: float = 0.0,
    crossover_prob: float = 0.7,
    mutation_prob: float = 0.1,
    avoid_inchikeys: Optional[set] = None,
    neighborhood_limit: int = 0
) -> List[str]:
    """
    Generate reaction candidates from Mol-Rxn-DB, optionally using genetic algorithm.
    Uses proper GA operators: selection, crossover, and mutation.
    
    Args:
        db_path: Path to molecules.sqlite database
        rxn_id: Reaction ID (4 or 5)
        num_candidates: Number of combinations to generate
        elite_names: List of elite molecule names for GA
        elite_scores: Optional dict mapping elite names to scores (for fitness-proportional selection)
        elite_frac: Fraction of candidates to generate from elites (0-1)
        crossover_prob: Probability of crossover between parents (0-1)
        mutation_prob: Mutation probability for GA (0-1)
        avoid_inchikeys: Set of InChIKeys to avoid
        neighborhood_limit: Neighborhood expansion limit for elites
        
    Returns:
        List of reaction strings: ["rxn:4:123:456:789", ...]
    """
    try:
        mols_A, mols_B, mols_C = get_reaction_molecule_pools(db_path, rxn_id)
        
        # Check if we have minimum required components (A and B are required, C may be optional)
        if not (mols_A and mols_B):
            bt.logging.warning(f"Not enough molecules for reaction {rxn_id}. A:{len(mols_A)}, B:{len(mols_B)}, C:{len(mols_C) if mols_C else 0}")
            return []
        
        # If C is required but missing, skip this reaction
        if not mols_C:
            bt.logging.debug(f"Reaction {rxn_id} missing C component, skipping")
            return []
        
        candidates = []
        
        # Use GA if elites are provided
        if elite_names and elite_frac > 0:
            n_elite = max(0, min(num_candidates, int(num_candidates * elite_frac)))
            n_rand = num_candidates - n_elite
            
            # Generate elite-based offspring using proper GA operators
            if n_elite > 0:
                elite_candidates = generate_offspring_from_elites(
                    rxn_id=rxn_id,
                    n=n_elite,
                    elite_names=elite_names,
                    elite_scores=elite_scores,
                    molecules_A=mols_A,
                    molecules_B=mols_B,
                    molecules_C=mols_C,
                    crossover_prob=crossover_prob,
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


async def _process_candidate_async(
    candidate: str, 
    config: argparse.Namespace, 
    weekly_target: str,
    loop: asyncio.AbstractEventLoop,
    seen_inchikeys: OrderedDict,
    seen_lock: asyncio.Lock
) -> Optional[Tuple[str, str, Optional[str]]]:
    """
    Process a single candidate asynchronously (non-blocking).
    Returns (name, canonical_smiles, inchikey) or None if invalid.
    """
    try:
        # Run blocking operations in executor
        smiles = await loop.run_in_executor(None, get_smiles_cached, candidate)
        if not smiles:
            return None
        
        # Canonicalize SMILES for consistent key matching
        canon_smiles = await loop.run_in_executor(None, canonicalize_smiles, smiles)
        if not canon_smiles:
            return None
        
        # Validate molecule (blocking call in executor)
        is_valid = await loop.run_in_executor(
            None, 
            validate_molecule_cached,
            canon_smiles, 
            weekly_target, 
            config.min_heavy_atoms, 
            config.min_rotatable_bonds, 
            config.max_rotatable_bonds
        )
        if not is_valid:
            return None
        
        # Get InChIKey (blocking call in executor)
        inchikey = await loop.run_in_executor(None, smiles_to_inchikey_cached, canon_smiles)
        
        # Only skip if we've already scored this exact molecule (thread-safe check)
        if inchikey:
            async with seen_lock:
                if inchikey in seen_inchikeys:
                    return None
        
        return (candidate, canon_smiles, inchikey)
    except Exception as e:
        bt.logging.debug(f"Error processing candidate {candidate}: {e}")
        return None


# Split validation: static checks (cacheable) vs weekly-target checks (less cacheable)
@lru_cache(maxsize=100_000)
def validate_molecule_static_cached(smiles: str, min_heavy_atoms: int, min_rotatable_bonds: int, max_rotatable_bonds: int) -> bool:
    """
    Cached validation of static molecule properties (independent of weekly target).
    This improves cache hits since these checks don't depend on weekly_target.
    
    Args:
        smiles: SMILES string (should be canonical)
        min_heavy_atoms: Minimum heavy atoms
        min_rotatable_bonds: Minimum rotatable bonds
        max_rotatable_bonds: Maximum rotatable bonds
        
    Returns:
        bool: True if static properties are valid, False otherwise
    """
    try:
        # Check heavy atoms
        if get_heavy_atom_count(smiles) < min_heavy_atoms:
            return False
        
        # Check rotatable bonds (using cached mol)
        mol = mol_from_smiles_cached(smiles)
        if mol is None:
            return False
        
        num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        if num_rotatable_bonds < min_rotatable_bonds or num_rotatable_bonds > max_rotatable_bonds:
            return False
        
        return True
        
    except Exception as e:
        bt.logging.debug(f"Static validation error for {smiles}: {e}")
        return False


@lru_cache(maxsize=50_000)
def validate_molecule_cached(smiles: str, weekly_target: str, min_heavy_atoms: int, min_rotatable_bonds: int, max_rotatable_bonds: int) -> bool:
    """
    Cached validation of a molecule (cached by all parameters).
    Uses split validation: static checks first (better cache hits), then weekly-target check.
    
    Args:
        smiles: SMILES string (should be canonical)
        weekly_target: Target protein code
        min_heavy_atoms: Minimum heavy atoms
        min_rotatable_bonds: Minimum rotatable bonds
        max_rotatable_bonds: Maximum rotatable bonds
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # First check static properties (better cache hits)
        if not validate_molecule_static_cached(smiles, min_heavy_atoms, min_rotatable_bonds, max_rotatable_bonds):
            return False
        
        # Then check uniqueness for protein (depends on weekly_target, less cacheable)
        if not molecule_unique_for_protein_hf(weekly_target, smiles):
            return False
        
        return True
        
    except Exception as e:
        bt.logging.debug(f"Validation error for {smiles}: {e}")
        return False


def validate_molecule(smiles: str, config: argparse.Namespace, weekly_target: str) -> bool:
    """
    Validate a molecule meets all requirements (wrapper for cached function).
    
    Args:
        smiles: SMILES string
        config: Configuration object
        weekly_target: Target protein code
        
    Returns:
        bool: True if valid, False otherwise
    """
    return validate_molecule_cached(
        smiles, 
        weekly_target, 
        config.min_heavy_atoms, 
        config.min_rotatable_bonds, 
        config.max_rotatable_bonds
    )


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
    loop = asyncio.get_running_loop()
    
    bt.logging.info("Starting Boltz model inference loop.")
    
    # Initialize Boltz model per-task (not global) to avoid race conditions
    # Use lock to serialize init/cleanup and avoid race conditions
    if 'boltz_lock' not in state:
        state['boltz_lock'] = asyncio.Lock()
    
    async with state['boltz_lock']:
        if state.get('boltz') is None:
            bt.logging.info("Initializing Boltz model...")
            state['boltz'] = BoltzWrapper()
            bt.logging.info("Boltz model initialized successfully")
        boltz = state['boltz']
    
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
    # Configurable batch size (default optimized for RTX 4090)
    batch_size = getattr(config, 'batch_size', None) or 128
    sleep_time = getattr(config, 'sleep_time', None) or 0.1
    candidates_per_batch = 2000  # Generate 2000 candidates at a time (1000 per reaction) - MAXIMIZED for RTX 4090
    
    # Elite pool management - larger for better diversity
    elite_pool_size = 500  # Keep top 500 candidates - MAXIMIZED for RTX 4090
    elite_pool = pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
    
    # LRU cache for seen_inchikeys to prevent unbounded growth
    MAX_SEEN_INCHIKEYS = 200_000
    seen_inchikeys = OrderedDict()
    seen_lock = asyncio.Lock()  # Lock for thread-safe access to seen_inchikeys
    
    async def mark_seen_inchikey(key: str) -> None:
        """Mark an InChIKey as seen, with LRU eviction (async, thread-safe)."""
        if key:
            async with seen_lock:
                seen_inchikeys[key] = True
                if len(seen_inchikeys) > MAX_SEEN_INCHIKEYS:
                    seen_inchikeys.popitem(last=False)
    
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
            # Get elite names and scores for GA (fitness-proportional selection)
            elite_names = elite_pool["name"].tolist() if not elite_pool.empty else None
            elite_scores = None
            if not elite_pool.empty:
                elite_scores = dict(zip(elite_pool["name"], elite_pool["score"]))
            
            # Crossover probability (canonical GA typically uses 0.6-0.9)
            crossover_prob = 0.7
            
            # Generate new candidates on-demand (with proper GA operators)
            # Pass OrderedDict directly (membership check is O(1) and thread-safe for reads)
            # No need to snapshot - OrderedDict supports 'in' operator efficiently
            async with seen_lock:
                # Snapshot keys under lock to avoid race conditions during iteration
                avoid_inchikeys_set = set(seen_inchikeys.keys()) if seen_inchikeys else set()
            
            candidates_rxn4 = get_reaction_candidates(
                db_path, rxn_id=4, num_candidates=candidates_per_batch // 2,
                elite_names=elite_names, elite_scores=elite_scores, elite_frac=elite_frac,
                crossover_prob=crossover_prob, mutation_prob=mutation_prob, 
                avoid_inchikeys=avoid_inchikeys_set, neighborhood_limit=neighborhood_limit
            )
            candidates_rxn5 = get_reaction_candidates(
                db_path, rxn_id=5, num_candidates=candidates_per_batch // 2,
                elite_names=elite_names, elite_scores=elite_scores, elite_frac=elite_frac,
                crossover_prob=crossover_prob, mutation_prob=mutation_prob, 
                avoid_inchikeys=avoid_inchikeys_set, neighborhood_limit=neighborhood_limit
            )
            all_candidates = candidates_rxn4 + candidates_rxn5
            
            if not all_candidates:
                bt.logging.warning("No candidates generated, waiting...")
                await asyncio.sleep(sleep_time)  # Configurable sleep time
                continue
            
            # Process in batches - accumulate for larger GPU batches
            processed = 0
            accumulated_valid = []  # Accumulate valid molecules for larger scoring batches
            accumulated_names = []
            accumulated_inchikeys = []
            # Dynamic accumulation target (lower threshold to prevent stalling)
            accumulation_target = batch_size * 2  # Reduced from 4x to 2x to prevent stalling
            min_accumulation = batch_size  # Minimum before scoring (prevents infinite wait)
            
            while not state['shutdown_event'].is_set() and processed < len(all_candidates):
                # Get next batch
                batch_end = min(processed + batch_size, len(all_candidates))
                batch = all_candidates[processed:batch_end]
                processed = batch_end
                
                # Validate and convert to SMILES (using cached functions)
                valid_molecules = []
                valid_names = []
                valid_inchikeys = []
                
                # Process batch asynchronously to avoid blocking event loop
                batch_tasks = []
                for candidate in batch:
                    batch_tasks.append(_process_candidate_async(
                        candidate, config, weekly_target, loop, seen_inchikeys, seen_lock
                    ))
                
                # Wait for all candidates in batch to be processed
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        continue
                    if result is None:
                        continue
                    name, canon_smiles, inchikey = result
                    valid_molecules.append(canon_smiles)
                    valid_names.append(name)
                    valid_inchikeys.append(inchikey)
                
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
                # Also score if we have minimum and yield is low (prevents infinite accumulation)
                has_minimum = len(accumulated_valid) >= min_accumulation
                has_target = len(accumulated_valid) >= accumulation_target
                is_end = processed >= len(all_candidates)
                # Dynamic: if yield is very low, lower threshold
                yield_ratio = len(accumulated_valid) / max(processed, 1)
                low_yield = yield_ratio < 0.05 and has_minimum  # Less than 5% yield
                
                should_score = has_target or is_end or low_yield
                
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
                    # Run Boltz scoring in executor to avoid blocking event loop
                    await loop.run_in_executor(
                        None,
                        boltz.score_molecules_target,
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
                            # Boltz may return scores keyed by canonical SMILES or InChIKey
                            # Try canonical SMILES first (most common), then InChIKey as fallback
                            for idx, canon_smiles in enumerate(valid_molecules):
                                mol_score = per_molecule_metric[0].get(canon_smiles)
                                if mol_score is None and idx < len(valid_inchikeys) and valid_inchikeys[idx]:
                                    mol_score = per_molecule_metric[0].get(valid_inchikeys[idx])
                                
                                if mol_score is not None and math.isfinite(mol_score):
                                    batch_scores.append({
                                        "name": valid_names[idx],
                                        "smiles": canon_smiles,
                                        "InChIKey": valid_inchikeys[idx] if idx < len(valid_inchikeys) else None,
                                        "score": mol_score
                                    })
                        else:
                            # Use average score for all molecules
                            for idx, canon_smiles in enumerate(valid_molecules):
                                batch_scores.append({
                                    "name": valid_names[idx],
                                    "smiles": canon_smiles,
                                    "InChIKey": valid_inchikeys[idx] if idx < len(valid_inchikeys) else None,
                                    "score": boltz_score
                                })
                        
                        if batch_scores:
                            # Update seen InChIKeys (using LRU eviction, async)
                            for item in batch_scores:
                                if item["InChIKey"]:
                                    await mark_seen_inchikey(item["InChIKey"])
                            
                            # Merge with elite pool, deduplicate keeping highest score, and take top N
                            # Note: For very large pools (>10k), consider using a dict keyed by InChIKey
                            # for O(1) updates instead of repeated concat + sort + drop_duplicates
                            batch_df = pd.DataFrame(batch_scores)
                            elite_pool = pd.concat([elite_pool, batch_df], ignore_index=True)
                            # Fix: Sort by score DESC first, then drop duplicates keeping first (highest score)
                            elite_pool = elite_pool.sort_values(by="score", ascending=False)
                            elite_pool = elite_pool.drop_duplicates(subset=["InChIKey"], keep="first")
                            elite_pool = elite_pool.head(elite_pool_size).reset_index(drop=True)
                            
                            # Update best candidate (top of elite pool)
                            if not elite_pool.empty:
                                best_row = elite_pool.iloc[0]
                                best_score = best_row["score"]
                                best_candidate = best_row["name"]
                                
                                if best_score > state['best_score']:
                                    state['best_score'] = best_score
                                    state['candidate_product'] = best_candidate
                                    bt.logging.info(f"New best score: {best_score:.4f}, Candidate: {best_candidate}, Elite pool size: {len(elite_pool)}")
                            
                            # Adaptive GA parameters with bounds (prevent parameter explosion)
                            if iteration > 5 and not elite_pool.empty:  # Start adapting after more iterations
                                current_mean = elite_pool["score"].mean()
                                
                                # Adjust based on duplicate ratio - prioritize exploration in competition
                                # Refined calculation: use nunique to get actual duplicate count
                                unique_in_batch = batch_df["InChIKey"].nunique() if not batch_df.empty else 0
                                dup_ratio = (len(batch_scores) - unique_in_batch) / max(1, len(batch_scores))
                                
                                # Periodic review: prevent too aggressive mutation/exploration that reduces convergence
                                # Only adapt if we have enough data (prevents early over-adaptation)
                                if iteration > 10:  # Review after more iterations
                                    if dup_ratio > 0.5:  # Lower threshold - more aggressive exploration
                                        mutation_prob = min(0.4, mutation_prob * 1.05)  # More conservative: reduced multiplier, lower cap
                                        elite_frac = max(0.1, elite_frac * 0.95)  # Slower decrease
                                        neighborhood_limit = min(6, neighborhood_limit + 1)  # Lower cap to prevent explosion
                                    elif dup_ratio < 0.15 and not elite_pool.empty:
                                        mutation_prob = max(0.15, mutation_prob * 0.99)  # More conservative decrease
                                        elite_frac = min(0.25, elite_frac * 1.01)  # More conservative increase, lower cap
                                
                                # Adjust based on score improvement - more exploration-focused but with convergence checks
                                if prev_mean_score is not None and iteration > 10:
                                    score_improvement = max(0, (current_mean - prev_mean_score) / max(abs(prev_mean_score), 1e-6))
                                    if score_improvement < 0.03:  # Lower threshold - explore more aggressively
                                        # Explore more - competition needs diversity (with bounds)
                                        mutation_prob = min(0.4, mutation_prob * 1.05)  # More conservative
                                        elite_frac = max(0.1, elite_frac * 0.97)  # Slower decrease
                                        neighborhood_limit = min(6, neighborhood_limit + 1)  # Lower cap
                                    elif score_improvement > 0.25:  # Only exploit if significant improvement
                                        # Exploit more - but still maintain some exploration (with bounds)
                                        mutation_prob = max(0.15, mutation_prob * 0.99)  # More conservative decrease
                                        elite_frac = min(0.25, elite_frac * 1.01)  # More conservative increase, lower cap
                                
                                prev_mean_score = current_mean
                            
                            # Log exploration stats periodically (more frequent for RTX 4090)
                            if iteration % 5 == 0:  # More frequent logging
                                elapsed = time.time() - start_time
                                # Read len under lock to avoid race conditions
                                async with seen_lock:
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
                    # Cache current_block to reduce API calls
                    if 'cached_block' not in state or 'block_cache_time' not in state or time.time() - state['block_cache_time'] > 5:
                        state['cached_block'] = await state['subtensor'].get_current_block()
                        state['block_cache_time'] = time.time()
                    current_block = state['cached_block']
                    
                    next_epoch_block = ((current_block // state['epoch_length']) + 1) * state['epoch_length']
                    blocks_until_epoch = next_epoch_block - current_block
                    
                    # Only minimal sleep if we have plenty of time (RTX 4090 can handle continuous load)
                    if blocks_until_epoch > 100:
                        await asyncio.sleep(sleep_time)  # Configurable sleep time
                    # No sleep if close to epoch end - maximize exploration
                except Exception:
                    await asyncio.sleep(sleep_time)  # Configurable fallback sleep
            
        except Exception as e:
            bt.logging.error(f"Error in Boltz model loop: {e}")
            traceback.print_exc()
            await asyncio.sleep(sleep_time)  # Configurable error recovery sleep
    
    bt.logging.info("Boltz model loop completed.")
    
    # Cleanup: free GPU memory and remove boltz from state (with lock)
    if 'boltz_lock' in state:
        async with state['boltz_lock']:
            if state.get('boltz') is not None:
                try:
                    if hasattr(state['boltz'], 'cleanup_model'):
                        state['boltz'].cleanup_model()
                except Exception:
                    pass
                del state['boltz']
                state['boltz'] = None
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass


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
    
    # 1) Encrypt the response (use cached block if available)
    if 'cached_block' in state and 'block_cache_time' in state:
        if time.time() - state['block_cache_time'] <= 5:
            current_block = state['cached_block']
        else:
            state['cached_block'] = await state['subtensor'].get_current_block()
            state['block_cache_time'] = time.time()
            current_block = state['cached_block']
    else:
        current_block = await state['subtensor'].get_current_block()
        state['cached_block'] = current_block
        state['block_cache_time'] = time.time()
    
    encrypted_response = state['bdt'].encrypt(state['miner_uid'], candidate_product, current_block)
    bt.logging.info(f"Encrypted response generated successfully")

    # 2) Create temp file, write content (Windows-compatible: delete=False, unlink manually)
    tmp_path = None
    try:
        tmp_file = tempfile.NamedTemporaryFile(delete=False, mode='w+')
        tmp_path = tmp_file.name
        tmp_file.close()  # Close immediately after getting path
        
        with open(tmp_path, 'w+') as f:
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

        # 3) Attempt chain commitment with retry/backoff
        bt.logging.info(f"Attempting chain commitment...")
        commitment_status = False
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try: 
                commitment_status = await state['subtensor'].set_commitment(
                    wallet=state['wallet'],
                    netuid=state['config'].netuid,
                    data=commit_content
                )
                bt.logging.info(f"Chain commitment status: {commitment_status}")
                break  # Success, exit retry loop
            except MetadataError:
                bt.logging.info("Too soon to commit again. Will keep looking for better candidates.")
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    bt.logging.warning(f"Chain commitment attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    bt.logging.error(f"Chain commitment failed after {max_retries} attempts: {e}")
                    return

        # 4) If chain commitment success, upload to GitHub with retry/backoff
        if commitment_status:
            max_github_retries = 3
            base_github_delay = 2.0
            
            for attempt in range(max_github_retries):
                try:
                    bt.logging.info(f"Commitment set successfully for {commit_content}")
                    bt.logging.info(f"Attempting GitHub upload (attempt {attempt + 1}/{max_github_retries})...")
                    github_status = upload_file_to_github(filename, encoded_content)
                    if github_status:
                        bt.logging.info(f"File uploaded successfully to {commit_content}")
                        state['last_submitted_product'] = candidate_product
                        state['last_submission_time'] = datetime.datetime.now()
                        break  # Success, exit retry loop
                    else:
                        if attempt < max_github_retries - 1:
                            delay = base_github_delay * (2 ** attempt)
                            bt.logging.warning(f"GitHub upload attempt {attempt + 1} failed. Retrying in {delay}s...")
                            await asyncio.sleep(delay)
                        else:
                            bt.logging.error(f"Failed to upload file to GitHub for {commit_content} after {max_github_retries} attempts")
                except Exception as e:
                    if attempt < max_github_retries - 1:
                        delay = base_github_delay * (2 ** attempt)
                        bt.logging.warning(f"GitHub upload error (attempt {attempt + 1}): {e}. Retrying in {delay}s...")
                        await asyncio.sleep(delay)
                    else:
                        bt.logging.error(f"Failed to upload file for {commit_content} after {max_github_retries} attempts: {e}")
    finally:
        # Clean up temp file (Windows-compatible)
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


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
    # Cache current_block to reduce API calls
    cached_block = None
    block_cache_time = 0
    block_cache_ttl = 5  # Cache for 5 seconds
    
    while True:
        try:
            # Use cached block if still valid, otherwise fetch new one
            # Initialize on first use to prevent None error
            if cached_block is None or time.time() - block_cache_time > block_cache_ttl:
                cached_block = await subtensor.get_current_block()
                block_cache_time = time.time()
            current_block = cached_block

            # If we are at an epoch boundary, fetch new proteins
            if current_block % epoch_length == 0:
                bt.logging.info(f"Found epoch boundary at block {current_block}.")
                
                # Calculate block hash once (avoid duplicate calculation)
                final_block_hash = await subtensor.determine_block_hash(current_block)
                block_hash = final_block_hash
                
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
