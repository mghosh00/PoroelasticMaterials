from setuptools import setup, find_packages


def get_version():
    """
    Get version number from the nonlinear_poroweakening module.
    """
    import os
    import sys

    sys.path.append(os.path.abspath('nonlinear_poroelasticity'))
    from version_info import VERSION as version
    sys.path.pop()

    return version


def get_requirements():
    requirements = []
    with open("requirements.txt", "r") as file:
        for line in file:
            requirements.append(line)
    return requirements


setup(
    # Module name
    name='nonlinear_poroelasticity',

    # Version
    version=get_version(),

    description='Finite-element based nonlinear poroelastic simulations',

    maintainer='Matthew Ghosh',

    maintainer_email='matthew.ghosh@gtc.ox.ac.uk',

    url='https://github.com/mghosh00/PoroelasticMaterials',

    # Packages to include
    packages=find_packages(include=('nonlinear_poroelasticity', 'nonlinear_poroelasticity.*')),

    # List of dependencies
    install_requires=get_requirements(),

    extras_require={
        'docs': [
            'sphinx>=1.5, !=1.7.3',
        ],
        'dev': [
            'flake8>=3',
            'pytest',
            'pytest-cov',
        ],
    },
)
