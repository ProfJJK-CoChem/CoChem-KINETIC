class MasterEquationGatekeeper:
    def __init__(self, max_memory_gb=32.0):
        self.max_memory_gb = max_memory_gb

    def request_diagonalization(self, N: int, num_grid_points: int):
        # dense N x N matrix of float64 (8 bytes)
        bytes_per_matrix = N * N * 8
        total_bytes = bytes_per_matrix * num_grid_points
        total_gb = total_bytes / (1024**3)

        if total_gb > self.max_memory_gb:
            return {
                "status": "Dry-Run",
                "reason": f"OOM limit exceeded. Storing dense {N}x{N} matrices for {num_grid_points} grid points requires {total_gb:.2f} GB RAM, which exceeds the {self.max_memory_gb} GB physical capacity.",
                "proposal": "Explicit permission required."
            }
        return {
            "status": "Allowed",
            "total_allocation_gb": total_gb
        }

class VTSTOptimizationGatekeeper:
    def __init__(self, max_node_hours=100.0):
        self.max_node_hours = max_node_hours

    def request_vtst_optimization(self, num_irc_points: int, level_of_theory: str, force: bool = False):
        # Rough estimation of node-hours per single point based on level of theory
        cost_per_point = {
            "DFT": 0.5,
            "CCSD(T)": 50.0
        }
        
        per_point = cost_per_point.get(level_of_theory.upper(), 1.0)
        total_node_hours = num_irc_points * per_point

        if not force and total_node_hours > self.max_node_hours:
            return {
                "status": "Dry-Run",
                "reason": f"Compute budget limit exceeded. {num_irc_points} points at {level_of_theory} require ~{total_node_hours:.1f} node-hours, exceeding the {self.max_node_hours} hr limit.",
                "proposal": "Explicit permission required (use force=True) or fallback to DFT."
            }
        
        return {
            "status": "Allowed",
            "total_node_hours": total_node_hours
        }
