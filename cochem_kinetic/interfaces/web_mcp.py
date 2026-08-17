import json
import numpy as np

class WebMCPArrheniusPlotter:
    def __init__(self) -> None:
        pass

    def generate_plot_json(self, T_array: list[float], k_array: list[float]) -> str:
        """
        Generates a JSON payload containing standard Plotly configuration for an Arrhenius plot.
        Ensures logarithmic scaling and correct inverse temperature rendering.
        """
        # We assume the UI layer expects Plotly JSON schema format.
        
        # Convert T_array to 1000/T
        try:
            inverse_t = [1000.0 / float(t) for t in T_array]
        except ZeroDivisionError as e:
            raise ValueError("Temperature array cannot contain zero") from e
        except (TypeError, ValueError) as e:
            raise TypeError("Temperature array must contain numeric values") from e
        
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
        
        try:
            return json.dumps(plot_data)
        except TypeError as e:
            raise ValueError("Failed to serialize plot data to JSON") from e
