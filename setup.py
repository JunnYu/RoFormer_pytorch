from setuptools import setup, find_packages

setup(
    name='roformer',
    package_dir={"": "src"},
    packages=find_packages("src"),
    version='0.0.1',
    license='MIT',
    description='roformer_pytorch',
    author='Jun Yu',
    author_email='573009727@qq.com',
    url='https://github.com/JunnYu/RoFormer_pytorch',
    keywords=['roformer', 'pytorch'],
    install_requires=['transformers>=4.5.0', 'jieba'],
)