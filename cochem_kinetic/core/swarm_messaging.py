from typing import Dict, Any

class VTSTOptimizationGatekeeper:
    def __init__(self, max_node_hours: float = 100.0):
        self.max_node_hours = max_node_hours
        
    def _estimate_cost(self, num_irc_points: int, level_of_theory: str, basis_functions: int = 100) -> float:
        """
        Estimates the computational cost in node-hours using physical scaling logic.
        CCSD(T) scales as O(N^7) where N is the number of basis functions.
        DFT scales roughly as O(N^3) to O(N^4).
        """
        # Baseline scaling factor to map to hours for a reference system (N=100)
        # Assuming for N=100, DFT takes ~0.1 hours
        # CCSD(T) takes ~5 hours
        N = basis_functions
        
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
            
        return float(num_irc_points * cost_per_point)
        
    def request_vtst_optimization(self, num_irc_points: int, level_of_theory: str, force: bool = False, basis_functions: int = 100) -> Dict[str, Any]:
        """
        Gates the VTST optimization payload.
        """
        estimated_cost = self._estimate_cost(num_irc_points, level_of_theory, basis_functions)
        
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
