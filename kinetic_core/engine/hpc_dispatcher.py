import asyncio
import json
import logging
import numpy as np
from typing import Any

logger = logging.getLogger(__name__)

class KineticHPCDispatcher:
    def __init__(self, batch_size: int = 1000) -> None:
        self.batch_size: int = batch_size
        self.swarm_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def _worker(self) -> None:
        while True:
            payload = await self.swarm_queue.get()
            try:
                payload_json = json.dumps(payload)
                cmd = ["sbatch", "--parsable", "--wrap", f"python -m kinetic_core.worker '{payload_json}'"]
                
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await process.communicate()
                
                if process.returncode != 0:
                    logger.error(f"Execution failed: Slurm submission failed with code {process.returncode}: {stderr.decode().strip()}")
            except FileNotFoundError:
                logger.error("Execution failed: sbatch command not found. Slurm is required for HPC dispatch.")
            except json.JSONDecodeError as e:
                logger.error(f"Execution failed: Failed to serialize payload: {e}")
            except Exception as e:
                logger.error(f"Execution failed: Unexpected error in HPC dispatch: {e}")
            finally:
                self.swarm_queue.task_done()

    async def dispatch_tp_grid(self, temperatures: Any, pressures: Any) -> tuple[list[dict[str, Any]], list[asyncio.Task[None]]]:
        """
        Slices the independent T/P coordinates into discrete /goal JSON payloads 
        and pushes them asynchronously to avoid freezing the node.
        """
        # Create full grid of coordinates flattened (10,000 points)
        T, P = np.meshgrid(temperatures, pressures)
        t_flat = T.flatten()
        p_flat = P.flatten()
        
        # Slicing into payloads asynchronously rather than nested loops
        payloads: list[dict[str, Any]] = [
            {"endpoint": "/goal", "temperature": float(t), "pressure": float(p)}
            for t, p in zip(t_flat, p_flat)
        ]
        
        num_workers = min(len(payloads), 1000)
        workers: list[asyncio.Task[None]] = [asyncio.create_task(self._worker()) for _ in range(num_workers)]
        
        # Pushing asynchronously 
        for payload in payloads:
            self.swarm_queue.put_nowait(payload)
            
        return payloads, workers

    async def dispatch_reaction_network(self, reactions: list[Any]) -> tuple[list[dict[str, Any]], list[asyncio.Task[None]]]:
        """
        Slices the elementary reactions into discrete /goal JSON payloads 
        and pushes them asynchronously to avoid freezing the node.
        """
        payloads: list[dict[str, Any]] = [
            {"endpoint": "/goal", "reaction_id": i, "reaction_data": r}
            for i, r in enumerate(reactions)
        ]
        
        num_workers = min(len(payloads), 1000)
        workers: list[asyncio.Task[None]] = [asyncio.create_task(self._worker()) for _ in range(num_workers)]
        
        # Pushing asynchronously 
        for payload in payloads:
            self.swarm_queue.put_nowait(payload)
            
        return payloads, workers

