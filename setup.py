from distutils.core import setup
import setuptools

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
  name = 'jax-random_projections',
  packages = setuptools.find_packages(),
  version = '1.0',
  license='MIT',
  description = 'sklearn\'s random projection with JAX to run on a GPU',
  long_description=long_description,
  long_description_content_type='text/markdown',
  author = 'Sebastian Bischoff',
  author_email = 'sebastian@salzreute.de',
  url = 'https://github.com/Baschdl/jax-random_projections',
  download_url = 'https://github.com/Baschdl/jax-random_projections/archive/v1.0.1.tar.gz',
  keywords = ['random projections', 'jax', 'GPU'],
  install_requires=[
          'sklearn',
          'jaxlib',
          'jax',
      ],
  classifiers=[
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
    'Topic :: Software Development',
    'Topic :: Scientific/Engineering',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Environment :: GPU',
  ],
)