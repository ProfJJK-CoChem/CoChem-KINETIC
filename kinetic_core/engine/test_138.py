import asyncio
import sys
import os

# Add the repo root to sys.path so kinetic_core can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from kinetic_core.engine.hpc_dispatcher import KineticHPCDispatcher

async def main():
    dispatcher = KineticHPCDispatcher()
    
    # 1. Generate combustion network of 1000 elementary steps
    reactions = [f"reaction_{i}" for i in range(1000)]
    
    # 2. Dispatch the network
    payloads, workers = await dispatcher.dispatch_reaction_network(reactions)
    
    # 3. Assert all 1000 payloads are pushed to Slurm simultaneously
    # In KineticHPCDispatcher, they are pushed to dispatcher.swarm_queue
    print(f"Total payloads created: {len(payloads)}")
    print(f"Items in swarm queue: {dispatcher.swarm_queue.qsize()}")
    
    assert len(payloads) == 1000, f"Expected 1000 payloads, got {len(payloads)}"
    assert dispatcher.swarm_queue.qsize() == 1000, f"Expected 1000 items in queue, got {dispatcher.swarm_queue.qsize()}"
    
    # Clean up tasks
    for w in workers:
        w.cancel()

if __name__ == "__main__":
    asyncio.run(main())
