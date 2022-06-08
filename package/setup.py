from codecs import open
from os.path import abspath, dirname, join
from subprocess import call

from setuptools import Command, find_packages, setup

from thesispack import __version__

this_dir = abspath(dirname(__file__))
with open(join(this_dir, 'README.md'), encoding='utf-8') as file:
    long_description = file.read()


long_description = """
    
"""

setup(
    name='thesispack',
    version=__version__,
    description='Thesis package files.',
    long_description=long_description,
    url='https://github.com/efth-mcl/indoor-localization-with-ml',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha'
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    packages=['thesispack'],
    package_dir={'thesispack': 'thesispack'},
    install_requires=[
        'pydocstyle',
        'tensorflow',
        'numpy',
        'pandas',
        'matplotlib',
        'scikit-learn',
        'networkx'
    ],
)

