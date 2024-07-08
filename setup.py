from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        subprocess.call(['bash', 'post_install.sh'])

setup(
    name='autoalign',
    version='0.0.1',
    description='',
    author='luxinyu1',
    author_email='luxinyu2021@iscas.ac.cn',
    python_requires='>=3.10',
    packages=find_packages(),
    install_requires=[
        'torch==2.3.0',
        'tensorboard',
        'wandb',
        'datasets',
        'tokenizers',
        'sentencepiece',
        'numpy',
        'pandas',
        'transformers>=4.41.2',
        'deepspeed',
        'pydantic',
        'packaging',
        'accelerate',
        'ninja',
        'einops',
        'wandb',
        'fire',
        'trl>=0.9.3',
    ],
    extras_require={
        'train': [
            'flash-attn>=2.0',
        ],
        'eval': [
            'torchvision==0.18.0',
            'torchaudio==2.3.0',
            'vllm==0.4.3',
            'human-eval==1.0.3',
            'alpaca-eval==0.6.3',
            'opencompass==0.2.3',
        ],
        'dev': [
            'pytest',
            'pre-commit',
        ],
    },
    cmdclass={
        'install': PostInstallCommand,
    },
)