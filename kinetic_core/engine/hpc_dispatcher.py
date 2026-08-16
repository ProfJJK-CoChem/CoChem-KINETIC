import asyncio
import numpy as np

class KineticHPCDispatcher:
    def __init__(self, batch_size=1000):
        self.batch_size = batch_size
        self.swarm_queue = asyncio.Queue()

    async def _worker(self):
        while True:
            payload = await self.swarm_queue.get()
            try:
                raise NotImplementedError("[ERR_MISSING_BIN] Real /goal payload evaluation not implemented.")
            finally:
                self.swarm_queue.task_done()

    async def dispatch_tp_grid(self, temperatures, pressures):
        """
        Slices the independent T/P coordinates into discrete /goal JSON payloads 
        and pushes them asynchronously to avoid freezing the node.
        """
        # Create full grid of coordinates flattened (10,000 points)
        T, P = np.meshgrid(temperatures, pressures)
        t_flat = T.flatten()
        p_flat = P.flatten()
        
        # Slicing into payloads asynchronously rather than nested loops
        payloads = [
            {"endpoint": "/goal", "temperature": float(t), "pressure": float(p)}
            for t, p in zip(t_flat, p_flat)
        ]
        
        num_workers = min(len(payloads), 1000)
        workers = [asyncio.create_task(self._worker()) for _ in range(num_workers)]
        
        # Pushing asynchronously 
        for payload in payloads:
            self.swarm_queue.put_nowait(payload)
            
        return payloads, workers

    async def dispatch_reaction_network(self, reactions):
        """
        Slices the elementary reactions into discrete /goal JSON payloads 
        and pushes them asynchronously to avoid freezing the node.
        """
        payloads = [
            {"endpoint": "/goal", "reaction_id": i, "reaction_data": r}
            for i, r in enumerate(reactions)
        ]
        
        num_workers = min(len(payloads), 1000)
        workers = [asyncio.create_task(self._worker()) for _ in range(num_workers)]
        
        # Pushing asynchronously 
        for payload in payloads:
            self.swarm_queue.put_nowait(payload)
            
        return payloads, workers

