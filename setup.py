from setuptools import setup, find_packages

setup(
    name="damage-prediction-ae",
    version="0.1.0",
    description="",
    author="Yacine Bel-Hadj, Francisco de Nolasco Santos",
    author_email="yacine.bel-hadj@vub.be, francisco.de.nolasco.santos@vub.be",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
)
