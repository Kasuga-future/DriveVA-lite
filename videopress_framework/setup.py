from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent

setup(
    name="driveva-videotokenpress",
    version="0.1.0",
    description="Independent VideoTokenPress framework for DriveVA token experiments.",
    packages=find_packages(),
    python_requires=">=3.10",
)
