from setuptools import find_packages, setup

setup(
    name="roformer",
    package_dir={"": "src"},
    packages=find_packages("src"),
    version="0.2.0",
    license="Apache 2.0",
    description="roformer_pytorch",
    author="Jun Yu",
    author_email="573009727@qq.com",
    url="https://github.com/JunnYu/RoFormer_pytorch",
    keywords=["roformer", "pytorch", "tf2.0"],
    install_requires=["transformers>=4.7.0", "jieba"],
)
