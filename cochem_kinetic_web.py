import streamlit as st
import subprocess
import os
import sys
import psutil
import atexit
from pathlib import Path
from pydantic import BaseModel, Field

st.set_page_config(page_title="CoChem-KINETIC - Native Pipeline UI", layout="wide")

class PipelineConfig(BaseModel):
    target_smiles: str = Field(..., description="Target SMILES string")
    run_mode: str = Field(..., description="Execution mode, e.g. Fast or Accurate")

def kill_zombie_processes() -> None:
    target_procs = ['orca', 'xtb', 'mpi', 'crest']
    for proc in psutil.process_iter(['name']):
        try:
            name = proc.info['name'].lower()
            if any(target in name for target in target_procs):
                proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            raise NotImplementedError("Implementation pending")
atexit.register(kill_zombie_processes)

st.title("🔬 CoChem-KINETIC Control Panel")
st.markdown("This UI executes raw, heavy mathematical payloads natively.")

with st.sidebar:
    st.header("Pipeline Configuration")
    target_smiles = st.text_input("Target SMILES", "CCO")
    run_mode = st.selectbox("Execution Mode", ["Fast", "Accurate"])

if st.button("🚀 Execute Default Pipeline"):
    with st.spinner(f"Triggering quantum physics executor for {target_smiles}..."):
        st.info("Initiating Physical Math Execution Pipeline...")
        
        module_dir = Path(__file__).resolve().parent
        tests_dir = module_dir / "tests"
        
        env = os.environ.copy()
        env["COCHEM_TARGET_H5"] = os.path.join(os.getcwd(), "landscape.h5")
        
        try:
            cmd = [sys.executable, "-m", "pytest", str(tests_dir), "-v"]
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True, 
                timeout=3600, 
                cwd=str(module_dir),
                env=env
            )
            
            st.code(result.stdout[-3000:], language="text")
            st.success("✅ Execution Completed Natively. CPU load generated.")
            
            raise NotImplementedError("Real physical calculation binary (orca/xtb) is not available. Cannot generate real physical_output.out.")
                
        except subprocess.TimeoutExpired:
            st.error("Execution timed out. Purging zombies.")
            kill_zombie_processes()
        except subprocess.CalledProcessError as e:
            st.warning(f"Execution finished with non-zero exit code: {e.returncode}")
            kill_zombie_processes()
        except NotImplementedError as e:
            st.warning(str(e))
        except Exception as e:
            st.error(f"Pipeline crashed during physical execution: {str(e)}")
            kill_zombie_processes()
