
"""Packaging settings."""

from codecs import open
from os.path import abspath, dirname, join
from subprocess import call

from setuptools import Command, find_packages, setup

from thesispack import __version__

this_dir = abspath(dirname(__file__))
with open(join(this_dir, 'README.md'), encoding='utf-8') as file:
    long_description = file.read()


long_description = """
    There put efthymis-mcl thesis package files.
"""

setup(
    name='thesispack',
    version=__version__,
    description='Thesis package files.',
    long_description=long_description,
    url='https://github.com/mmlab-aueb/efthymis-mcl-MSc-Thesis',
    license='UNLICENSE',
    classifiers=[
        'Development Status :: 3 - Alpha'
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    # keywords = 'cli',
    packages=['thesispack'],
    package_dir={'thesispack': 'thesispack'},
    install_requires=[
        # 'docopt',
        "pydocstyle",
        'tensorflow==2.2.0',
        'numpy==1.19.1',
        'pandas==1.0.5',
        'matplotlib==3.3.0',
        'tensorboard==2.2.2',
        'scikit-learn==0.23.1',
        'spektral==0.6.1'
    ],
    # entry_points={
    #     'console_scripts': [
    #         'thesispack = thesispack.cli:main'
    #     ],
    # },
    test_suite='nose.collector',
    tests_require=['nose'],
    # extras_require = {
    #     'test': ['coverage', 'pytest', 'pytest-cov'],
    # },
)

