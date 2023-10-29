from setuptools import setup

setup(
    name='torch-fem',
    version='0.1',
    description='A Finite Element Method library for PyTorch',
    url='https://github.com/yourusername/my_package',
    author='walker chi',
    author_email='walker.chi.000@gmail.com',
    license='MIT',
    packages=['torch_fem'],
    install_requires=[
        'torch',
        'meshio',
        'cupy',
        'pyvista',
        'numpy',
        'scipy',
        'matplotlib'
    ],
    zip_safe=False
)
