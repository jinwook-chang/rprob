# setup.py

from setuptools import setup, find_packages

setup(
    name='rprob',
    version='1.0.4',
    description='A library to use R style probability distribution functions in Python',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Jinwook Chang',
    author_email='tranquil_morningl@icloud.com',
    url='https://github.com/jinwook-chang/rprob',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
