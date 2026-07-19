from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="turing_pattern_simulator", 
    version="1.0.0",                 
    author="Your Name",              
    author_description="Simulation of Turing Patterns (Leopard and Giraffe)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    packages=find_packages(), 

    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
    ],

    entry_points={
        'console_scripts': [
            'turing-sim=turing_core.main:main', 
        ],
    },

    python_requires='>=3.7',
)