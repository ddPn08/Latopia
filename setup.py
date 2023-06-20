from setuptools import setup

setup(
    name="latopia",
    packages=["latopia"],
    version="0.0.1",
    url="https://github.com/ddPn08/Latopia",
    description="Speech AI training and inference tools",
    author="ddPn08",
    author_email="contact@ddpn.world",
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "latopia=latopia.cli.main:main",
        ]
    },
)
