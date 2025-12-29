import os
import sys
import subprocess
from setuptools import setup
from setuptools.command.install import install

class CustomInstall(install):
    def run(self):
        install.run(self)

        # Check the value of INSTALL_VLLM_ASCEND environment variable
        install_vllm_ascend = os.environ.get('INSTALL_VLLM_ASCEND', '0')  # Default to '0' if not set
        if install_vllm_ascend not in ('0', '1'):
            raise ValueError(
                f"Invalid value for INSTALL_VLLM_ASCEND: {install_vllm_ascend!r}. "
                "Expected '0' or '1'."
            )

        if install_vllm_ascend == '1':
            print("Installing vLLM from source...")
            env = os.environ.copy()
            env['VLLM_TARGET_DEVICE'] = 'empty'  # Optional: Define specific env variable if needed

            # Install vLLM
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install',
                'git+https://github.com/vllm-project/vllm.git@v0.7.3',  
                '--extra-index-url', 'https://download.pytorch.org/whl/cpu/'
            ], env=env)

            print("Installing vLLM Ascend from source...")
            # Install vLLM-Ascend
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install',
                'git+https://github.com/vllm-project/vllm-ascend.git@v0.7.3.post1', 
                '--extra-index-url', 'https://download.pytorch.org/whl/cpu/'
            ])
        else:
            print("Skipping vLLM and vLLM Ascend installation...")
            print("Installing vLLM version 0.8.3 instead...")
            # Install the specific version of vLLM (vllm==0.8.3)
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install',
                'vllm==0.8.3'
            ])

setup(cmdclass={'install': CustomInstall})
