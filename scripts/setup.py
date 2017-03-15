from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = [
    "keras==1.2.2",
    "theano",
    "h5py",
]

setup(
    name='dataflow_preprocess',
    version='1.0',
    description='Image prepocessing code to be run on Google Dataflow',
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
)
