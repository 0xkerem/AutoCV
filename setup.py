from setuptools import setup, find_packages
import autocv

# Read the contents of the README file
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="AutoCV",
    version=autocv.__version__,
    author=autocv.__author__,
    author_email=autocv.__email__,
    description="An automated cross-validation framework for machine learning models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/0xkerem/AutoCV",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.9',
    install_requires=[
        "numpy",
        "scikit-learn",
    ],
    entry_points={
        'console_scripts': [
            'autocv=autocv.core:main',
        ],
    },
)
