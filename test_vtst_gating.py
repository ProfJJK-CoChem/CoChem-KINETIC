import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
import sys
import os

# Add the CoChem-KINETIC to the path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from kinetic_core.swarm_messaging import VTSTOptimizationGatekeeper

def main():
    logger.info("Initializing VTST Optimization Gatekeeper...")
    gatekeeper = VTSTOptimizationGatekeeper(max_node_hours=100.0)
    
    # 1. The VTST Bomb Test: Request a canonical VTST optimization spanning 100 IRC points with highly accurate CCSD(T)
    logger.info("Dispatching massive VTST payload: 100 IRC points with CCSD(T)...")
    payload_response = gatekeeper.request_vtst_optimization(
        num_irc_points=100,
        level_of_theory="CCSD(T)",
        force=False
    )
    
    # 2. Gatekeeper Audit & 3. Block Verification
    logger.info(f"Gatekeeper Response: {payload_response}")
    
    assert payload_response["status"] == "Dry-Run", f"Failed: Status is {payload_response['status']}, expected Dry-Run"
    assert "Explicit permission required" in payload_response["proposal"] or "fallback to DFT" in payload_response["proposal"], "Failed: Did not propose explicit permission or fallback to DFT"
    
    logger.info("SUCCESS: Execution correctly halted and the Dry-Run payload flagged a critical budget violation.")

if __name__ == "__main__":
    main()
