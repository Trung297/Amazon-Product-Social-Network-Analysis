from setuptools import setup, find_packages

setup(
    name="community-detection-project",
    version="1.0.0",
    description="Graph analysis and community detection on Amazon co-purchasing network",
    author="Viet",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scipy",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "networkx",
        "streamlit",
    ],
    python_requires=">=3.8",
)