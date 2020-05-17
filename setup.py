from distutils.core import setup
import setuptools

setup(
  name = 'jax-random_projections',
  packages = setuptools.find_packages(),
  version = '1.0',
  license='MIT',
  description = 'sklearn\'s random projection with JAX to run on a GPU',
  author = 'Sebastian Bischoff',
  author_email = 'sebastian@salzreute.de',
  url = 'https://github.com/Baschdl/jax-random_projections',
  download_url = 'https://github.com/Baschdl/jax-random_projections/archive/v1.0.tar.gz',
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