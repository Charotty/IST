from setuptools import setup, find_packages

setup(
    name="its_project",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "ccxt>=4.0.0",
        "aiohttp>=3.8.0",
    ],
    python_requires=">=3.8",
)
