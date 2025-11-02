import os
import sys
import subprocess
from setuptools import setup
from setuptools.command.install import install

class CustomInstall(install):
    def run(self):
        install.run(self)

        print("Installing vLLM from source...")
        env = os.environ.copy()
        env['VLLM_TARGET_DEVICE'] = 'empty'
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install',
            'git+https://github.com/vllm-project/vllm.git@v0.7.3',
            '--extra-index-url', 'https://download.pytorch.org/whl/cpu/'
        ], env=env)

        print("Installing vLLM Ascend from source...")
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install',
            'git+https://github.com/vllm-project/vllm-ascend.git@v0.7.3.post1',
            '--extra-index-url', 'https://download.pytorch.org/whl/cpu/'
        ])

setup(cmdclass={'install': CustomInstall})