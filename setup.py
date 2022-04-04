#!/usr/bin/env python3
from setuptools import find_packages, setup


def setup_asymmetricselfplay():
    setup(
        name="asymmetricselfplay",
        version=open("ASYMMETRICSELFPLAY_VERSION").read(),
        packages=find_packages(),
        install_requires=[
        
        ],
        python_requires=">=3.7.4",
        description="Paper implementation of Asymmetric self-play for automatic goal discovery in robotic manipulation",
        include_package_data=True,
    )


setup_asymmetricselfplay()