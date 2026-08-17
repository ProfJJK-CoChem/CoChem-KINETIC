from typing import Any

class MasterEquationGatekeeper:
    def __init__(self, max_memory_gb: float = 32.0) -> None:
        self.max_memory_gb: float = max_memory_gb

    def request_diagonalization(self, N: int, num_grid_points: int) -> dict[str, Any]:
        try:
            if N <= 0 or num_grid_points <= 0:
                raise ValueError("N and num_grid_points must be positive integers.")
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
        except (TypeError, ValueError) as e:
            return {
                "status": "Error",
                "reason": f"Invalid input parameters: {e}"
            }

class VTSTOptimizationGatekeeper:
    def __init__(self, max_node_hours: float = 100.0) -> None:
        self.max_node_hours: float = max_node_hours

    def request_vtst_optimization(self, num_irc_points: int, level_of_theory: str, force: bool = False) -> dict[str, Any]:
        try:
            if num_irc_points <= 0:
                raise ValueError("num_irc_points must be a positive integer.")
            if not isinstance(level_of_theory, str):
                raise TypeError("level_of_theory must be a string.")
            
            # Rough estimation of node-hours per single point based on level of theory
            cost_per_point: dict[str, float] = {
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
        except (TypeError, ValueError, AttributeError) as e:
            return {
                "status": "Error",
                "reason": f"Invalid input parameters: {e}"
            }
