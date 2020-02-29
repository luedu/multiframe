#!/usr/bin/env python

import runpy
from setuptools import setup


def read_requires_file(name):
    """Read a requires-*.txt file

    These are like requirements.txt files, but since we're not pinning versions
    we import them into setup.py and they don't have all the special syntax that
    requirements files support.

    The main reason for these is to permit better layering/caching in the
    docker images.
    """
    with open('requires-{}.txt'.format(name), 'r') as file:
        lines = (line.strip() for line in file)
        return [line for line in lines if line and not line.startswith('#')]


SETUP = dict(
    name='multiframe',
    version='1.0.0',
    description='Multiframe Pipeline',
    long_description="",
    author='Luis García',
    author_email='luisgarciac@outlook.es',
    install_requires=read_requires_file('install'),
    include_package_data=True,
    python_requires='>=3.6'
)

if __name__ == '__main__':
    setup(**SETUP)
