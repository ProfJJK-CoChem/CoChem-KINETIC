#!/usr/bin/env python3
"""
CoChem-KINETIC - Stage 4: Interactive 3D HTML IRC Animation Generator
----------------------------------------------------------------------
Renders 3D py3Dmol / Plotly HTML trajectory animations of IRC pathways for Jupyter UI.
"""

import hashlib
import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class IRCAnimationVisualizer:
    """Generates interactive 3D HTML animations of IRC pathways."""

    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir or Path(os.environ.get("COCHEM_ARTIFACT_DIR", "."))

    def build_multi_xyz_string(self, symbols: list[str], trajectory_coords: np.ndarray) -> str:
        """
        Converts 3D trajectory array (N_frames, N_atoms, 3) to Multi-XYZ string.
        """
        xyz_lines = []
        n_atoms = len(symbols)
        for frame_idx, coords in enumerate(trajectory_coords):
            xyz_lines.append(f"{n_atoms}")
            xyz_lines.append(f"IRC Frame {frame_idx + 1}")
            for sym, (x, y, z) in zip(symbols, coords):
                xyz_lines.append(f"{sym:2s} {x:12.6f} {y:12.6f} {z:12.6f}")
        return "\n".join(xyz_lines)

    def generate_html_animation(self, reaction_name: str, symbols: list[str], trajectory_coords: np.ndarray) -> str:
        """
        Generates a standalone HTML document embedding a 3D py3Dmol animation viewer.
        """
        multi_xyz = self.build_multi_xyz_string(symbols, trajectory_coords)
        multi_xyz_escaped = multi_xyz.replace("`", "'").replace("\n", "\\n")

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>CoChem-KINETIC 3D IRC Animation: {reaction_name}</title>
    <script src="https://3dmol.org/build/3Dmol-min.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; margin: 20px; background-color: #1e1e1e; color: #eee; }}
        #container {{ width: 800px; height: 500px; position: relative; border: 1px solid #444; margin: 0 auto; border-radius: 8px; overflow: hidden; }}
        h2 {{ text-align: center; color: #61afef; }}
    </style>
</head>
<body>
    <h2>CoChem-KINETIC 3D IRC Trajectory: {reaction_name}</h2>
    <div id="container"></div>
    <script>
        var xyzData = `{multi_xyz_escaped}`;
        var viewer = $3Dmol.createViewer("container", {{ backgroundColor: "black" }});
        viewer.addModelsAsFrames(xyzData, "xyz");
        viewer.setStyle({{}}, {{ sphere: {{ radius: 0.4 }}, stick: {{ radius: 0.15 }} }});
        viewer.zoomTo();
        viewer.animate({{ loop: "backAndForth", interval: 100 }});
        viewer.render();
    </script>
</body>
</html>
"""
        out_file = self.output_dir / f"{reaction_name}_irc_animation.html"
        try:
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(html_content)
        except PermissionError as e:
            logger.error(f"Permission denied writing HTML animation: {e}")
            raise
        except OSError as e:
            logger.error(f"OS error writing HTML animation: {e}")
            raise
        return html_content


def calculate_artifact_sha256(filepath: str | Path) -> str:
    """Calculates SHA-256 hash of a computational artifact."""
    p = Path(filepath)
    hasher = hashlib.sha256()
    try:
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
    except FileNotFoundError as e:
        logger.error(f"Artifact file not found: {filepath}")
        raise
    except OSError as e:
        logger.error(f"OS error calculating hash for {filepath}: {e}")
        raise
    return hasher.hexdigest()