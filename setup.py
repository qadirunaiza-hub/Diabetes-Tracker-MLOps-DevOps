from setuptools import find_packages, setup

setup(
    name="diabetes-tracker",
    version="1.0.0",
    description="ML-based diabetes prediction and monitoring system",
    author="Mu-Sigma",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.1.0",
        "numpy>=1.26.0",
        "scikit-learn>=1.4.0",
        "xgboost>=2.0.0",
        "mlflow>=2.10.0",
        "flask>=3.0.0",
        "flask-cors>=4.0.0",
        "gunicorn>=21.2.0",
        "pyyaml>=6.0.1",
        "requests>=2.31.0",
        "joblib>=1.3.0",
    ],
)
