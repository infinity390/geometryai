from setuptools import setup, find_packages

setup(
    name="geometryai",
    version="0.1.2",
    description="solve euclid geometry",
    author="Your Name",
    author_email="you@example.com",
    packages=find_packages(),
    python_requires=">=3.8",

    install_requires=[
        "mathai"
    ],
)