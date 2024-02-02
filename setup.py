from setuptools import setup, find_packages

setup(
    name='rl_trajectory',
    version='0.0.1',
    packages=find_packages(),
    url='https://github.com/elizabethnewman/rl_trajectory.git',
    license='MIT',
    author='Elizabeth Newman',
    author_email='elizabeth.newman@emory.edu',
    description='',
    install_requires=['gym==0.26.2', 'numpy==1.24.3', 'torch==2.0.1', 'matplotlib'],
    python_requires='>=3.6'
)
