"""
Created on Thu Oct 25 14:28:56 2018

@author: joshualamstein
"""

from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['h5py==2.8.0']

#Get versions of packages

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Gedi Project'
)
