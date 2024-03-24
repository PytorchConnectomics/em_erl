import os, sys
from setuptools import setup, find_packages


def setup_package():
    __version__ = '0.1'
    url = 'https://github.com/PytorchConnectomics/em-erl'
    setup(name='em_erl',
        description='Expected run length (ERL) evaluation',
        version=__version__,
        url=url,
        license='MIT',
        author='Donglai Wei',
        install_requires=['argparse', 'em_util'],
        python_requires='>3.7',
        packages=find_packages()
    )

if __name__=='__main__':
    # pip install --editable .
    setup_package()
