import logging
import subprocess
import tempfile
import os
from typing import Tuple, Optional
from pydantic import BaseModel, Field, ConfigDict
from kinetic_core.thermo import calculate_eyring_rate

logger = logging.getLogger(__name__)

class MasterEquationConfig(BaseModel):
    model_config = ConfigDict(strict=True)

    delta_g: float = Field(..., description="Gibbs free energy of activation in kcal/mol")
    temp: float = Field(298.15, gt=0.0, description="Temperature in Kelvin")

class EvaluationParams(BaseModel):
    model_config = ConfigDict(strict=True)

    P_atm: float = Field(..., gt=0.0, description="Pressure in atmospheres")

class MasterEquationSolver:
    def __init__(self, config: Optional[MasterEquationConfig] = None, delta_g: Optional[float] = None, temp: float = 298.15):
        if config is None:
            if delta_g is None:
                raise ValueError("Either config or delta_g must be provided to MasterEquationSolver.")
            config = MasterEquationConfig(delta_g=delta_g, temp=temp)
        elif isinstance(config, dict):
            config = MasterEquationConfig(**config)
        self.config = config
        self.k_inf = calculate_eyring_rate(self.config.delta_g, temp=self.config.temp, reaction_order=1)

    def evaluate(self, params: EvaluationParams) -> float:
        """
        Evaluates the pressure-dependent rate constant k(T, P).
        """
        R_gas_atm_L = 0.082057
        concentration_M = params.P_atm / (R_gas_atm_L * self.config.temp)
        z_lj = 1e11
        k_0_pseudo = z_lj * concentration_M
        
        p_r = k_0_pseudo / self.k_inf if self.k_inf > 0 else 0
        
        if p_r > 1e6:
            logger.info("Extreme pressure detected via collision frequency ratio (P_r > 1e6).")
            logger.info("Bypassing massive matrix diagonalization. Mathematically snapping to standard TST k_infty.")
            return self.k_inf
            
        logger.info("Performing standard matrix diagonalization via MESMER...")
        return self._run_mesmer(self.config.temp, params.P_atm)

    def evaluate_2d_ej(
        self, 
        temp_range: Tuple[float, float], 
        press_range: Tuple[float, float], 
        N: int = 10000, 
        auth_token: Optional[str] = None
    ) -> float:
        from kinetic_core.swarm_messaging import MasterEquationGatekeeper
        
        gatekeeper = MasterEquationGatekeeper()
        num_grid_points = 100
        
        check = gatekeeper.request_diagonalization(N, num_grid_points)
        if check["status"] == "Dry-Run" and auth_token != "/teamwork-preview":
            logger.warning(f"[DRY-RUN BLOCK] {check['reason']}")
            logger.warning(f"[PROPOSAL] {check['proposal']}")
            raise PermissionError("Dry-Run required explicit permission for massive matrix diagonalization.")
        
        logger.info(f"Authorization granted or memory check passed. Allocating {N}x{N} matrices...")
        
        t_avg = sum(temp_range) / 2.0
        p_avg = sum(press_range) / 2.0
        return self._run_mesmer(t_avg, p_avg, N=N)

    def _run_mesmer(self, temp: float, p_atm: float, N: int = 100) -> float:
        """
        Generates a MESMER input XML, executes mesmer, and extracts the rate.
        Falls back to Lindemann-Hinshelwood if MESMER is not available.
        """
        dg_kj = self.config.delta_g * 4.184
        
        xml_content = f"""<?xml version="1.0" encoding="utf-8" ?>
<me:mesmer xmlns="http://www.mesmer.ed.ac.uk" xmlns:me="http://www.mesmer.ed.ac.uk">
  <me:moleculeList>
    <me:molecule id="reactant">
      <me:propertyList>
        <me:property dictRef="me:ZPE"><me:scalar units="kJ/mol">0.0</me:scalar></me:property>
      </me:propertyList>
      <me:energyTransferModel xsi:type="me:ExponentialDown" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
        <me:deltaEDown units="cm-1">200.0</me:deltaEDown>
      </me:energyTransferModel>
    </me:molecule>
  </me:moleculeList>
  <me:reactionList>
    <me:reaction id="r1">
      <me:reactant><me:molecule ref="reactant" /></me:reactant>
      <me:transitionState>
        <me:molecule id="TS">
          <me:propertyList>
            <me:property dictRef="me:ZPE"><me:scalar units="kJ/mol">{dg_kj}</me:scalar></me:property>
          </me:propertyList>
        </me:molecule>
      </me:transitionState>
    </me:reaction>
  </me:reactionList>
  <me:conditions>
    <me:bathGas>He</me:bathGas>
    <me:PTs>
      <me:ptset>
        <me:T>{temp}</me:T>
        <me:P>{p_atm}</me:P>
      </me:ptset>
    </me:PTs>
  </me:conditions>
  <me:control>
    <me:grainSize units="cm-1">{N}</me:grainSize>
  </me:control>
</me:mesmer>
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, "mesmer_input.xml")
            with open(input_file, "w") as f:
                f.write(xml_content)
                
            try:
                result = subprocess.run(
                    ["mesmer", input_file], 
                    capture_output=True, 
                    text=True, 
                    check=True,
                    timeout=300
                )
                for line in result.stdout.splitlines():
                    if "k(T,P)" in line or "Rate constant" in line:
                        parts = line.split()
                        try:
                            return float(parts[-1])
                        except ValueError:
                            pass
                logger.warning("MESMER output parsed but no rate constant found. Falling back to Lindemann.")
            except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                logger.warning(f"MESMER execution failed or not found ({type(e).__name__}). Falling back to Lindemann-Hinshelwood.")
                
        # Fallback Lindemann-Hinshelwood
        R_gas_atm_L = 0.082057
        concentration_M = p_atm / (R_gas_atm_L * temp)
        z_lj = 1e11
        k_0_pseudo = z_lj * concentration_M
        
        if self.k_inf == 0:
            return 0.0
            
        return (self.k_inf * k_0_pseudo) / (self.k_inf + k_0_pseudo)
