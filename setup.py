from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="iq-determinism-critique",
    version="0.1.0",
    author="Savant Lab",
    author_email="savantlab@example.com",
    description="Research tools for analyzing probabilistic-to-deterministic transformations in IQ testing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/savantlab/iq-determinism-critique",
    packages=find_packages(exclude=['tests', 'examples']),
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "aiohttp>=3.8.0",
        "beautifulsoup4>=4.11.0",
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "scipy>=1.9.0",
        "jupyter>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "flake8>=5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "iq-critique-scrape=iq_critique.cli:scrape_main",
            "iq-critique-analyze=iq_critique.cli:analyze_main",
            "iq-critique-reading=iq_critique.cli:reading_main",
        ],
    },
)
