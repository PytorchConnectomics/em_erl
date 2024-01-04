import os, sys
from setuptools import setup, find_packages


def setup_package(no_cython=True):
    __version__ = '0.1'
    url = 'https://github.com/PytorchConnectomics/erl_eval'
    setup(name='erl_eval',
        description='Expected run length (ERL) evaluation',
        version=__version__,
        url=url,
        license='MIT',
        author='Donglai Wei',
        install_requires=['scipy','numpy','networkx','h5py','imageio', 'argparse'],
        packages=find_packages(),
    )

if __name__=='__main__':
    # pip install --editable .
    setup_package()
