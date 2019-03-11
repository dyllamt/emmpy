import os

from setuptools import setup, find_packages

module_dir = os.path.dirname(os.path.abspath(__file__))
setup(name='semitransport',
      version='0.1',
      description='analyze semiconductor transport data',
      author='Maxwell Dylla',
      license='MIT',
      packages=find_packages(),
      install_requires=['numpy', 'scipy', 'fdint'],
      long_description=open('readme.md').read())
