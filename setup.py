from setuptools import setup

requirements = [
    'sympy>=1.9',
    "vegas>=6.4.1",
]

setup(
    name="feynite",
    version="0.1.0",
    description="Tensor representation and Monte Carlo integration of Finite "
                "Feynman integrals: based on https://arxiv.org/pdf/2410.18014",
    url="",
    author="L. de la Cruz",
    packages=["feynite"],
    package_dir={"": "src"},
    install_requires=requirements,
    extras_require={
    },
)