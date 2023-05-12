from setuptools import setup, find_packages

setup(
    name='bthesis-transformer',
    version='0.1',
    description="'Transformers for Natural Language' implementation for Bachelor's thesis",
    author='Ryan Ott',
    author_email='ryanottofficial@gmail.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'fire'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)