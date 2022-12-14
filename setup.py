import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MLOps HW",
    author="Theodor Lisin",
    author_email="theo.lisin@gmail.com",
    description="MADE MLOps HW",
    keywords="Python, MADE, mlops",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/made-mlops-2022/mlops_LisinFedor",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    version="0.1.0",
    classifiers=[
        # see https://pypi.org/classifiers/
        "Development Status :: 1 - Alpha",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "matplotlib",
        "pandas",
        "numpy",
        "scikit-learn",
        "dvc[s3]",
        "marshmallow_dataclass",
        "mlflow",
        "fastapi",
        "uvicorn",
        "python-dotenv",
    ],
    extras_require={
        "dev": [
            "apache-airflow>=2.4.0",
            "apache-airflow-providers-docker==3.2.0",
            "apache-airflow-providers-amazon==2.4.0",
            "wemake-python-styleguide",
            "mypy",
            "black",
            "jupyterlab",
            "seaborn",
            "types-PyYAML",
        ],
        "tests": [
            "sdv",
            "pytest",
            "pytest-dotenv",
            "httpx",
        ],
    },
    entry_points={
        "console_scripts": [
            "classification = ml_project.__main__:main",
            "app = online_inference.__main__:main",
        ],
    },
)
