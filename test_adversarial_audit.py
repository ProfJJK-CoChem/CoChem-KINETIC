import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
import sys
import os

# Add local path to sys.path so we can import kinetic_core
sys.path.insert(0, os.path.abspath("."))

from kinetic_core.engine.master_equation_limits import MasterEquationSolver

def test_master_equation_sabotage():
    logger.info("--- Starting 2D E/J Master Equation Sabotage Test ---")
    solver = MasterEquationSolver(delta_g=15.0, temp=298.15)
    
    # Grid points for T from 100K to 2000K, P from 0.01 Torr to 100 atm
    # N is 10000 for a dense matrix
    temp_range = (100.0, 2000.0)
    press_range = (0.01 / 760.0, 100.0)
    
    try:
        solver.evaluate_2d_ej(temp_range, press_range, N=10000)
    except PermissionError as e:
        logger.info(f"Exception caught successfully: {e}")
        logger.info("--- Test Passed! Execution halted due to Dry-Run projection. ---")
        return
        
    logger.info("--- Test Failed! Execution proceeded without Dry-Run authorization! ---")

if __name__ == "__main__":
    test_master_equation_sabotage()
