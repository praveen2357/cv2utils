"""
/**
 * @author praveen kumar yalal
 * @email praveen2357@gmail.com
 * @desc [description]
 */
"""

import setuptools
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='cvutils',
    version='0.0.1',
    description='Opencv utilities',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/praveen2357/cvutils",
    author='praveen',
    author_email='praveen2357@gmail.com',
    license='MIT',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    zip_safe=False
)

