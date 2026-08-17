from typing import Any

class VTSTOptimizationGatekeeper:
    def __init__(self, max_node_hours: float = 100.0) -> None:
        try:
            self.max_node_hours: float = float(max_node_hours)
        except (ValueError, TypeError) as e:
            raise ValueError(f"max_node_hours must be a valid float: {e}") from e
        
    def _estimate_cost(self, num_irc_points: int, level_of_theory: str, basis_functions: int = 100) -> float:
        """
        Estimates the computational cost in node-hours using physical scaling logic.
        CCSD(T) scales as O(N^7) where N is the number of basis functions.
        DFT scales roughly as O(N^3) to O(N^4).
        """
        try:
            points = int(num_irc_points)
            N = int(basis_functions)
            if points < 0 or N <= 0:
                raise ValueError("Points must be >= 0 and basis functions > 0.")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid inputs for cost estimation: {e}") from e
            
        if "CCSD(T)" in level_of_theory:
            # O(N^7) scaling
            base_cost = 5.0
            scaling = (N / 100.0)**7
            cost_per_point = base_cost * scaling
        elif "DFT" in level_of_theory or "B3LYP" in level_of_theory:
            # O(N^3.5) average scaling for DFT
            base_cost = 0.1
            scaling = (N / 100.0)**3.5
            cost_per_point = base_cost * scaling
        else:
            # Default to some intermediate HF-like O(N^4) scaling
            base_cost = 1.0
            scaling = (N / 100.0)**4
            cost_per_point = base_cost * scaling
            
        return float(points * cost_per_point)
        
    def request_vtst_optimization(
        self, 
        num_irc_points: int, 
        level_of_theory: str, 
        force: bool = False, 
        basis_functions: int = 100
    ) -> dict[str, Any]:
        """
        Gates the VTST optimization payload.
        """
        try:
            estimated_cost = self._estimate_cost(num_irc_points, level_of_theory, basis_functions)
        except ValueError as e:
            return {
                "status": "Error",
                "message": str(e)
            }
        
        if estimated_cost > self.max_node_hours and not force:
            return {
                "status": "Dry-Run",
                "estimated_cost_hours": estimated_cost,
                "budget_limit": self.max_node_hours,
                "proposal": f"Explicit permission required (use force=True) or fallback to DFT. Cost ({estimated_cost:.2f} hrs) exceeds limit ({self.max_node_hours:.2f} hrs)."
            }
            
        return {
            "status": "Running",
            "estimated_cost_hours": estimated_cost,
            "message": "Optimization started."
        }
