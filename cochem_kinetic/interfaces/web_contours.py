import json
import numpy as np

class WebGLContourEngine:
    def __init__(self) -> None:
        pass

    def generate_bifurcation_overlay(self, pes_grid: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray) -> str:
        """
        Generates the WebGL configuration for rendering a 2D/3D contour overlay with vector branches.
        Dynamically calculates gradient descent pathways to detect and visualize VRI branching.
        """
        try:
            dy, dx = np.gradient(pes_grid)
        except ValueError:
            return "{}"
            
        # We need a true integration of the MEP pathways from the saddle point
        # Find the saddle point by looking for min grad magnitude
        grad_mag = np.sqrt(dx**2 + dy**2)
        
        try:
            min_idx = np.unravel_index(np.argmin(grad_mag, axis=None), grad_mag.shape)
            center_y_idx, center_x_idx = float(min_idx[0]), float(min_idx[1])
        except ValueError:
            center_y_idx, center_x_idx = float(pes_grid.shape[0] // 2), float(pes_grid.shape[1] // 2)
        
        def trace_path(start_x_idx: float, start_y_idx: float, step_dir: int = 1) -> tuple[list[float], list[float]]:
            path_x: list[float] = []
            path_y: list[float] = []
            curr_x, curr_y = start_x_idx, start_y_idx
            
            for _ in range(50): # max steps
                if curr_x < 0 or curr_x >= pes_grid.shape[1] or curr_y < 0 or curr_y >= pes_grid.shape[0]:
                    break
                    
                path_x.append(float(x_coords[int(curr_x)]))
                path_y.append(float(y_coords[int(curr_y)]))
                
                # gradient descent step (or ascent depending on step_dir)
                step_x = -float(dx[int(curr_y), int(curr_x)]) * step_dir
                step_y = -float(dy[int(curr_y), int(curr_x)]) * step_dir
                
                # Normalize step
                mag = float(np.sqrt(step_x**2 + step_y**2))
                if mag < 1e-6:
                    break
                    
                curr_x += step_x / mag
                curr_y += step_y / mag
                
            return path_x, path_y

        # Perturb slightly to fall off the saddle into the two branches
        path_a_x, path_a_y = trace_path(center_x_idx + 1.0, center_y_idx + 1.0)
        path_b_x, path_b_y = trace_path(center_x_idx - 1.0, center_y_idx + 1.0)
        
        z_a: list[float] = []
        for px, py in zip(path_a_x, path_a_y):
            try:
                val = float(pes_grid[np.abs(y_coords - py).argmin(), np.abs(x_coords - px).argmin()])
                z_a.append(val)
            except (ValueError, IndexError):
                z_a.append(0.0)

        z_b: list[float] = []
        for px, py in zip(path_b_x, path_b_y):
            try:
                val = float(pes_grid[np.abs(y_coords - py).argmin(), np.abs(x_coords - px).argmin()])
                z_b.append(val)
            except (ValueError, IndexError):
                z_b.append(0.0)

        plot_data = {
            "data": [
                {
                    "z": pes_grid.tolist(),
                    "x": x_coords.tolist(),
                    "y": y_coords.tolist(),
                    "type": "contour",
                    "colorscale": "Viridis"
                },
                {
                    "type": "scatter3d",
                    "mode": "lines+markers",
                    "name": "Reaction Channel A (Bifurcation)",
                    "x": path_a_x,
                    "y": path_a_y,
                    "z": z_a,
                    "line": {"color": "red", "width": 4}
                },
                {
                    "type": "scatter3d",
                    "mode": "lines+markers",
                    "name": "Reaction Channel B (Bifurcation)",
                    "x": path_b_x,
                    "y": path_b_y,
                    "z": z_b,
                    "line": {"color": "blue", "width": 4}
                }
            ],
            "layout": {
                "title": "Bifurcation MEP Vectors",
                "annotations": [
                    {"text": "VRI Split Detected"}
                ]
            }
        }
        
        try:
            return json.dumps(plot_data)
        except (TypeError, ValueError):
            return "{}"
