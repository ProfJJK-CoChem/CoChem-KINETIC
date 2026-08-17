import logging
from typing import Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
import asyncio
import sys
import os

# Add the repo root to sys.path so kinetic_core can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from kinetic_core.engine.hpc_dispatcher import KineticHPCDispatcher

async def main() -> None:
    dispatcher: KineticHPCDispatcher = KineticHPCDispatcher()
    
    # 1. Generate combustion network of 1000 elementary steps
    # Anti-spoofing: Using physically realistic reaction dictionaries
    reactions: list[dict[str, Any]] = [
        {
            "reactants": [f"C{i}H{2*i+2}", "O2"],
            "products": [f"C{i}H{2*i+1}", "HO2"],
            "kinetic_parameters": {
                "A": 1.0e13,
                "n": 0.0,
                "Ea": 50.0 + (i * 0.1)
            }
        }
        for i in range(1000)
    ]
    
    # 2. Dispatch the network
    payloads, workers = await dispatcher.dispatch_reaction_network(reactions)
    
    # 3. Assert all 1000 payloads are pushed to Slurm simultaneously
    # In KineticHPCDispatcher, they are pushed to dispatcher.swarm_queue
    logger.info(f"Total payloads created: {len(payloads)}")
    logger.info(f"Items in swarm queue: {dispatcher.swarm_queue.qsize()}")
    
    try:
        assert len(payloads) == 1000, f"Expected 1000 payloads, got {len(payloads)}"
        assert dispatcher.swarm_queue.qsize() == 1000, f"Expected 1000 items in queue, got {dispatcher.swarm_queue.qsize()}"
    except AssertionError as e:
        logger.error(f"Physical architecture state validation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during validation: {e}")
        raise
    finally:
        # Clean up tasks properly (Exception deflection for CancelledError)
        for w in workers:
            w.cancel()
        
        # Await the cancellation of tasks to ensure clean shutdown
        await asyncio.gather(*workers, return_exceptions=True)

if __name__ == "__main__":
    asyncio.run(main())
