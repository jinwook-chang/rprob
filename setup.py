# setup.py

from setuptools import setup, find_packages

setup(
    name='rprob',  # 패키지 이름
    version='1.0.2',  # 버전
    description='A library to use R style probability distribution functions in Python',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Jinwook Chang',  # 본인의 이름
    author_email='tranquil_morningl@icloud.com',  # 본인의 이메일
    url='https://github.com/jinwook-chang/rprob',  # GitHub 등 저장소 URL
    license='MIT',  # 선택한 라이선스
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # 선택한 라이선스에 맞게 수정
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
