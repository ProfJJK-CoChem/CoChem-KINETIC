import json
import numpy as np

class WebMCPArrheniusPlotter:
    def __init__(self):
        raise NotImplementedError("Implementation pending")
    def generate_plot_json(self, T_array: list, k_array: list) -> str:
        """
        Generates a JSON payload containing standard Plotly configuration for an Arrhenius plot.
        Ensures logarithmic scaling and correct inverse temperature rendering.
        """
        # We assume the UI layer expects Plotly JSON schema format.
        
        # Convert T_array to 1000/T
        inverse_t = [1000.0 / t for t in T_array]
        
        plot_data = {
            "data": [
                {
                    "x": inverse_t,
                    "y": k_array,
                    "type": "scatter",
                    "mode": "lines+markers",
                    "name": "Rate Constant"
                }
            ],
            "layout": {
                "title": "Arrhenius Plot",
                "xaxis": {
                    "title": "1000 / T (K^-1)"
                },
                "yaxis": {
                    "title": "k(T)",
                    "type": "log"  # Crucial for non-linear Arrhenius curve rendering
                }
            }
        }
        
        return json.dumps(plot_data)
