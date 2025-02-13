import subprocess
from autoalign.paths import PROJ_BASE_PATH

def test_run_bash_script():
    result = subprocess.run(['bash', f'{PROJ_BASE_PATH}/scripts/test/test_core.sh'], 
                          capture_output=True,
                          text=True)
    
    assert result.returncode == 0