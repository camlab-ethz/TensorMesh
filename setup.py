import os
import re
from setuptools import setup, find_packages


def read_version():
    version_file = os.path.join(os.path.dirname(__file__), 'tensormesh', '_version.py')
    with open(version_file, 'r') as f:
        version_content = f.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_content, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name="tensormesh",
    version=read_version(),
    author="Shizheng Wen, Mingyuan Chi",
    author_email="shizheng.wen@sam.math.ethz.ch, walker.chi.000@gmail.com",
    description="Differentiable Finite Element Method Library for PyTorch",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    url="https://github.com/camlab-ethz/TensorMesh",
    project_urls={
        "Documentation": "https://camlab-ethz.github.io/TensorMesh/",
        "Source": "https://github.com/camlab-ethz/TensorMesh",
    },
    packages=find_packages(),
    install_requires=[
        "tqdm",
        "numpy",
        "scipy",
        "torch>=2.0.0",
        "torch-sla>=0.2.0",
        "meshio",
        "matplotlib",
        "psutil",
        "toml",
    ],
    extras_require={
        "petsc":[
            "petsc4py"
        ],
        "cupy":[
            "cupy"
        ],
        "example":[
            "plotly"
        ],
        "test": [
            "pytest",
            "pytest-cov",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords=[
        "deep-learning",
        "neural-networks",
        "AI4S",
        "ai-for-science",
        "pytorch",
        "numerical",
        "partial-differential-equation",
        "finite-element-methods",
        "fem",
        "gpu",
        "differentiable-simulation",
    ],
    python_requires=">=3.10",
)
