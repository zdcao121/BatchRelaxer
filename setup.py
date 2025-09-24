"""Setup for pip package."""
import unittest
from setuptools import setup
from setuptools import find_packages

REQUIRED_PACKAGES = [
    'orb-models',
    'ase<=3.25.0',
]

print(find_packages())

setup(
    name='BatchRelaxer',
    version='1.0',
    description='Batch relaxation using orb models',
    url='https://code.itp.ac.cn/zdcao/miscellaneous',
    author='zdcao',
    author_email='zdcao@iphy.ac.cn',
    # Contained modules and scripts.
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    extras_require={'testing': ['pytest']},
    platforms=['any'],
)
