from setuptools import find_packages, setup

setup(
    name="lifelong-drl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[],
    author="Kellen Kanarios",
    author_email="kellenkk@umich.edu",
    description="Applying streaming RL to lifelong learning",
    url="https://github.com/kkanarios32/lifelong-drl",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    py_modules=["optim", "sparse_init", "plot"],
)
