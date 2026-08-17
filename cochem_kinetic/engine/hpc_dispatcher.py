import asyncio
import json
from typing import Any

class KineticHPCDispatcher:
    def __init__(self) -> None:
        self.swarm_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        
    async def _evaluate_payload(self, payload: dict[str, Any]) -> str:
        """
        Evaluate the payload by genuinely dispatching it to an HPC job scheduling system (Slurm).
        """
        try:
            payload_json = json.dumps(payload)
            cmd = ["sbatch", "--parsable", "--wrap", f"python -m cochem_kinetic.worker '{payload_json}'"]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise RuntimeError(f"Slurm submission failed with code {process.returncode}: {stderr.decode().strip()}")
                
            return stdout.decode().strip()
            
        except FileNotFoundError:
            raise RuntimeError("sbatch command not found. Slurm is required for HPC dispatch.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to serialize payload: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error in HPC dispatch: {e}")
            
    async def dispatch_reaction_network(self, reactions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[asyncio.Task[str]]]:
        """
        Dispatches a massive array of reactions to the Slurm queue simultaneously.
        """
        payloads: list[dict[str, Any]] = []
        workers: list[asyncio.Task[str]] = []
        
        try:
            for i, reaction in enumerate(reactions):
                payload = {
                    "id": i,
                    "target": "/goal",
                    "reaction_data": reaction,
                    "task": "transition_state_rate_calculation"
                }
                payloads.append(payload)
                
            for payload in payloads:
                await self.swarm_queue.put(payload)
                
            for payload in payloads:
                task = asyncio.create_task(self._evaluate_payload(payload))
                workers.append(task)
                
        except asyncio.QueueFull:
            print("Error: The swarm queue is full.")
        except Exception as e:
            print(f"Error dispatching reaction network: {e}")
            
        return payloads, workers
        
    async def dispatch_tp_grid(self, T_grid: list[float], P_grid: list[float]) -> tuple[list[dict[str, Any]], list[asyncio.Task[str]]]:
        """
        Dispatches a 2D Temperature/Pressure grid asynchronously.
        Slices the grid into independent /goal payloads.
        """
        payloads: list[dict[str, Any]] = []
        workers: list[asyncio.Task[str]] = []
        
        try:
            idx = 0
            for T in T_grid:
                for P in P_grid:
                    payload = {
                        "id": idx,
                        "target": "/goal",
                        "task": "master_equation_grid_point",
                        "T": T,
                        "P": P
                    }
                    payloads.append(payload)
                    idx += 1
                    
            for payload in payloads:
                await self.swarm_queue.put(payload)
                
            for payload in payloads:
                task = asyncio.create_task(self._evaluate_payload(payload))
                workers.append(task)
                
        except asyncio.QueueFull:
            print("Error: The swarm queue is full.")
        except Exception as e:
            print(f"Error dispatching TP grid: {e}")
            
        return payloads, workers
