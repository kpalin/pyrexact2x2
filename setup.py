from setuptools import setup
import versioneer

requirements = [
    "pandas",
    "rpy2",
    # package requirements go here
]

setup(
    name='pyrexact2x2',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Python interface to R-package Exact2x2",
    license="MIT",
    author="Kimmo Palin",
    author_email='kimmo.palin@helsinki.fi',
    url='https://github.com/kpalin/pyrexact2x2',
    packages=['pyrexact2x2'],
    
    install_requires=requirements,
    keywords='pyrexact2x2',
    classifiers=[
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ]
)
