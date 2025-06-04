from setuptools import setup, find_packages

setup(
    name="nifty-options-analysis",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "duckdb",
        "pandas",
        "plotly",
        "numpy",
        "requests",
        "psutil",
        "streamlit-autorefresh"
    ],
) 