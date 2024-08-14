from setuptools import setup, find_packages

setup(
    name='prophet',
    version='0.1.0',
    url="https://github.com/theislab/prophet",
    license='CC-BY-NC 4.0',
    description='Scalable and universal prediction of cellular phenotypes',
    author='Alejandro Tejada-Lapuerta, Yuge Ji',
    author_email='alejandro.tejada@helmholtz-munich.de, yuge.ji@helmholtz-munich.de',
    packages=find_packages(),
    install_requires=[
        'joblib==1.4.2',
        'numpy==2.0.1',
        'pandas==1.5.3',
        'pytorch_lightning==2.1.0',
        'PyYAML==6.0.2',
        'scikit_learn==1.5.1',
        'scipy==1.14.0',
        'torch==2.3.0',
        'torchmetrics==1.4.0.post0',
        'tqdm==4.66.4',
        'wandb==0.17.6',
    ],
)