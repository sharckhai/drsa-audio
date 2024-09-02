from setuptools import setup, find_packages

setup(
    name='cxai',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'torchaudio',
        'torchvision',
        'librosa',
        'scipy',
        'matplotlib',
        'zennit',
    ],
    author='Samuel Harck',
    author_email='samuel.harck@icloud.com',
    description='Concept-based XAI on audio data',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/sharckhai/drsa-audio',
     classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
