from setuptools import setup, find_packages

setup(
    name='bgal_distribution',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy'
    ],
    description='BGAL Distribution Package',
    long_description='Package for fitting and testing the Bivariate Generalized Asymmetric Laplace (BGAL) distribution.',
    author='Cline',
    author_email='cline@example.com',
    license='MIT',
)
