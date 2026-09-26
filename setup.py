from setuptools import setup

setup(
    name="loss-landscape-analysis",
    version="1.3.0",
    description="Research project analyzing gradient behavior of MSE vs Cross-Entropy on MNIST",
    # NOTE: source lives in flat top-level modules (src/*.py), not packages,
    # so find_packages(where="src") resolves to []. Use py_modules instead.
    py_modules=[
        "data",
        "losses",
        "model",
        "trainer",
        "utils",
    ],
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.10",
)
