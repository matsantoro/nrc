from setuptools import setup

setup(
    name='nrc_package',
    version='0.0.1',
    author='Matteo Santoro',
    description='A module for handling reliability and tribe-based computations of experiments on connectomes.',
    packages=["nrc", "nrc.connectome", "nrc.reliability"],
)