from setuptools import setup, find_packages
from pathlib import Path

README = Path(__file__).parent / "README.md"

setup(
    name="geometryai",
    version="0.1.7",
    description="solve euclid geometry",
    long_description=README.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",

    author="educated_indian",
    author_email="you@example.com",

    packages=find_packages(),

    python_requires=">=3.8",

    install_requires=[
        "mathai"
    ],
)