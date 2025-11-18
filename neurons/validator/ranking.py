"""
Final scoring and winner determination functionality for the validator (Boltz-only)
"""

import math
import datetime
from typing import Optional

import bittensor as bt
from utils import calculate_dynamic_entropy


def calculate_final_scores(
    score_dict: dict[int, dict[str, any]],
    valid_molecules_by_uid: dict[int, dict[str, list[str]]],
    molecule_name_counts: dict[str, int],
    config: dict,
    current_epoch: int
) -> dict[int, dict[str, any]]:
    """
    Calculates final Boltz scores per UID, applying entropy bonus if applicable.
    
    Args:
        score_dict: Dictionary containing scores for each UID
        valid_molecules_by_uid: Dictionary of valid molecules by UID
        molecule_name_counts: Count of molecule name occurrences
        config: Configuration dictionary
        current_epoch: Current epoch number
        
    Returns:
        Updated score_dict with final scores calculated
    """
    
    dynamic_entropy_weight = calculate_dynamic_entropy(
        starting_weight=config['entropy_start_weight'],
        step_size=config['entropy_step_size'],
        start_epoch=config['entropy_start_epoch'],
        current_epoch=current_epoch
    )
    
    # Go through each UID scored
    for uid, data in valid_molecules_by_uid.items():
        boltz_score = score_dict[uid]['boltz_score']
        entropy_boltz = score_dict[uid]['entropy_boltz']
        threshold_boltz = config.get('entropy_bonus_threshold')

        # Apply entropy bonus if conditions are met
        if (
            boltz_score is not None
            and entropy_boltz is not None
            and math.isfinite(boltz_score)
            and math.isfinite(entropy_boltz)
            and boltz_score > threshold_boltz
            and entropy_boltz > 0
            and config['num_molecules_boltz'] > 1
        ):
            score_dict[uid]['boltz_score'] = boltz_score * (1 + (dynamic_entropy_weight * entropy_boltz))

        # Log details
        smiles_list = data.get('smiles', [])
        names_list = data.get('names', [])
        log_lines = [
            f"UID={uid}",
            f"  Molecule names: {names_list}",
            f"  SMILES: {smiles_list}",
            f"  Boltz scores: {score_dict[uid]['boltz_score']}",
        ]
        bt.logging.info("\n".join(log_lines))

    return score_dict


def determine_winner(score_dict: dict[int, dict[str, any]]) -> Optional[int]:
    """
    Determines the winning UID based on Boltz score.
    In case of ties, earliest submission time is used as the tiebreaker.
    
    Args:
        score_dict: Dictionary containing final scores for each UID
        
    Returns:
        Optional[int]: Winning UID or None if no valid scores found
    """
    best_score_boltz = -math.inf
    best_uids_boltz = []

    def parse_timestamp(uid):
        ts = score_dict[uid].get('push_time', '')
        try:
            return datetime.datetime.fromisoformat(ts)
        except Exception as e:
            bt.logging.warning(f"Failed to parse timestamp '{ts}' for UID={uid}: {e}")
            return datetime.datetime.max.replace(tzinfo=datetime.timezone.utc)

    def tie_breaker(tied_uids: list[int], best_score: float, model_name: str, print_message: bool = True):
        # Sort by block number first, then push time, then uid to ensure deterministic result
        winner = sorted(tied_uids, key=lambda uid: (
            score_dict[uid].get('block_submitted', float('inf')), 
            parse_timestamp(uid), 
            uid
        ))[0]
        
        winner_block = score_dict[winner].get('block_submitted')
        current_epoch = winner_block // 361 if winner_block else None
        push_time = score_dict[winner].get('push_time', '')
        
        tiebreaker_message = f"Epoch {current_epoch} tiebreaker {model_name} winner: UID={winner}, score={best_score}, block={winner_block}"
        if push_time:
            tiebreaker_message += f", push_time={push_time}"
            
        if print_message:
            bt.logging.info(tiebreaker_message)
            
        return winner
    
    # Find highest boltz_score
    for uid, data in score_dict.items():
        if 'boltz_score' not in data:
            continue
        
        boltz_score = round(data['boltz_score'], 4)
        
        if boltz_score > best_score_boltz:
            best_score_boltz = boltz_score
            best_uids_boltz = [uid]
        elif boltz_score == best_score_boltz:
            best_uids_boltz.append(uid)
    
    if not best_uids_boltz:
        bt.logging.info("No valid winner found (all scores -inf or no submissions).")
        return None

    # Treat all -inf as no valid winners
    if best_score_boltz == -math.inf:
        best_uids_boltz = []
    
    # Select winner
    if best_uids_boltz:
        if len(best_uids_boltz) == 1:
            boltz_winner_block = score_dict[best_uids_boltz[0]].get('block_submitted')
            current_epoch = boltz_winner_block // 361 if boltz_winner_block else None
            bt.logging.info(f"Epoch {current_epoch} BOLTZ winner: UID={best_uids_boltz[0]}, winning_score={best_score_boltz}")
            winner_boltz = best_uids_boltz[0]
        else:
            winner_boltz = tie_breaker(best_uids_boltz, best_score_boltz, "BOLTZ")
    else:
        winner_boltz = None
    
    return winner_boltz
