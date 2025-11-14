#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="Hiccup",
    version="2.0",
    description="A neural network potential trainer driven by gene algorithm",
    url="https://gitee.com/ccccissy/Hiccup",
    author="(1)Chen.Cheng, (2)Chen.Dingming",
    author_email="_cchen@mail.ecust.edu.cn",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'hiccup = core.main:main'
        ]
    },
    long_description="""A neural network potential trainer driven by gene algorithm""",
)
