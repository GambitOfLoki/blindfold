from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["ipython>=6", "nbformat>=4", "nbconvert>=5", "xgboost==1.6.0", "requests>=2", "pandas>=1.5"]

setup(
    name="blindfold",
    version="0.0.1",
    author="Yahya",
    author_email="iamgurudt@gmail.com",
    description="A package to implement LTR using XGBoost package",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/GambitOfLoki/blindfold",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.10.9",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)