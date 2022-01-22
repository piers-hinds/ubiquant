from setuptools import setup
from ubi.version import __version__

setup(
    name='ubi',
    url='https://github.com/piers-hinds/ubiquant',
    author='Piers Hinds',
    author_email='pmxph7@nottingham.ac.uk',
    packages=['ubi'],
    install_requires=['numpy', 'torch', 'pandas', 'sklearn', 'scipy'],
    tests_require=['pytest'],
    version=__version__,
    license='MIT',
    description='Helpers for Ubiquant Kaggle competition'
)