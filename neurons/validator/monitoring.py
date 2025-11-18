"""
Validator monitoring and reporting functionality
"""

import math
import os
import requests
import bittensor as bt


def monitor_validator(score_dict, metagraph, current_epoch, current_block, validator_hotkey, winning_uid):
    """
    Send validator monitoring data to external monitoring service.
    
    Args:
        score_dict: Dictionary of scores for all UIDs
        metagraph: Current metagraph
        current_epoch: Current epoch number
        current_block: Current block number
        validator_hotkey: Validator's hotkey address
        winning_uid: UID of the winning miner
    """
    api_key = os.environ.get('VALIDATOR_API_KEY')
    if not api_key:
        return
    
    try:
        import torch
        machine_info = {
            "torch_version": torch.__version__
        }
        if torch.cuda.is_available():
            machine_info["cuda_version"] = torch.version.cuda
            machine_info["gpu_name"] = torch.cuda.get_device_name(0)
        
        best_rounded_score = max([round(d['boltz_score'], 4) for d in score_dict.values() if 'boltz_score' in d and d['boltz_score'] is not None and math.isfinite(d['boltz_score'])], default=-math.inf)
        
        winning_group = []
        for uid, data in score_dict.items():
            boltz_score = data.get('boltz_score')
            if boltz_score is not None and math.isfinite(boltz_score) and round(boltz_score, 4) == best_rounded_score:
                winning_group.append({
                    "uid": uid,
                    "hotkey": metagraph.hotkeys[uid] if uid < len(metagraph.hotkeys) else "unknown",
                    "boltz_score": boltz_score,
                    "blocks_elapsed": (data.get('block_submitted', 0) % 361),
                    "push_time": data.get('push_time', ''),
                    "winner": uid == winning_uid
                })
        
        requests.post("https://valiwatch-production.up.railway.app/weights-info", json={
            "epoch": current_epoch,
            "current_block": current_block,
            "blocks_into_epoch": current_block % 361,
            "validator_hotkey": validator_hotkey,
            "validator_version": 1.5,  # Boltz-only version
            "winning_group": winning_group,
            "machine_info": machine_info
        }, headers={"Authorization": f"Bearer {api_key}"}, timeout=5)
        
    except Exception as e:
        bt.logging.debug(f"API send failed: {e}")
