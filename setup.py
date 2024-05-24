from setuptools import setup, find_packages

setup(
    name='mlmcbo',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'botorch==0.9.2',
        'numpy>=1.18.0',
        'matplotlib',
    ],
    python_requires='>=3.9',
    description='An extension of the BoTorch library adding new acquisition functions',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
    ],
)
