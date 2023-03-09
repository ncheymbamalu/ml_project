from setuptools import find_packages, setup
from typing import List


def get_packages(filepath: str) -> List[str]:
    """
    Returns a list of packages defined in requirements.txt
    """
    packages = [
        package 
        for package in open(filepath, "r").read().splitlines() 
        if package != "-e ."
    ]

    return packages


setup(
    name="ml_project", 
    version="0.0.1", 
    author="Nchey", 
    author_email="nchey.learnings@gmail.com", 
    packages=find_packages(), 
    install_requires=get_packages(r"requirements.txt")
)
