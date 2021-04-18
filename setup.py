from setuptools import setup, find_packages

setup(
    name='roformer',
    packages=find_packages(),
    version='0.0.1',
    license='MIT',
    description='roformer_pytorch',
    author='Jun Yu',
    author_email='573009727@qq.com',
    url='https://github.com/JunnYu/RoFormer_pytorch',
    keywords=['roformer', 'pytorch'],
    install_requires=['transformers'],
)