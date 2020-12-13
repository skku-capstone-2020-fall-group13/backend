  
from setuptools import setup, find_packages

setup(
    name="comflogic",
    version="1.0.0",
    author='Byeongseo Yu',
    author_email="qudtj1540@gmail.com",
    url="https://github.com/skku-capstone-2020-fall-group13/logic.git",
    install_requires= [
        'pillow>=2.0.0'
    ],
    description="Computation logic for comfort",
    packages=find_packages()
)
