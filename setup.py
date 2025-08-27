from setuptools import setup, find_packages

# Read long description from README
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "Scalable and universal prediction of cellular phenotypes"

setup(
    name="prophet",
    version="0.1.0",
    url="https://github.com/theislab/prophet",
    license="CC-BY-NC 4.0",
    description="Scalable and universal prediction of cellular phenotypes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Alejandro Tejada-Lapuerta, Yuge Ji",
    author_email="alejandro.tejada@helmholtz-munich.de, yuge.ji@helmholtz-munich.de",
    packages=find_packages(),
    python_requires=">=3.10",
    # All dependencies included - no extras needed
    install_requires=[
        # Core scientific computing
        "numpy>=2.0.0,<3.0.0",
        "pandas>=2.0.0,<3.0.0",
        "scikit-learn>=1.3.0,<2.0.0",
        "scipy>=1.10.0,<2.0.0",
        "tqdm>=4.60.0",
        "PyYAML>=6.0.0",
        "joblib>=1.3.0",
        # Deep learning framework
        "torch>=2.0.0,<3.0.0",
        "pytorch_lightning>=2.0.0,<3.0.0",
        "torchmetrics>=1.0.0,<2.0.0",
        # Visualization and notebooks
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "plotly>=5.0.0",
        "jupyterlab>=4.0.0",
        # Experiment tracking
        "wandb>=0.15.0",
        "tensorboard>=2.10.0",
        # Model downloading and sharing
        "huggingface_hub>=0.15.0",
    ],
    # Package metadata
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    # Keywords for package discovery
    keywords=[
        "machine learning",
        "biology",
        "drug discovery",
        "cell biology",
        "perturbations",
        "transformer",
        "predictions",
    ],
    # Include package data
    include_package_data=True,
)
