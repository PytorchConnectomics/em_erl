from setuptools import setup, find_packages


def setup_package():
    __version__ = "0.1"
    url = "https://github.com/PytorchConnectomics/em_erl"
    setup(
        name="em_erl",
        description="Expected run length (ERL) evaluation",
        version=__version__,
        url=url,
        license="MIT",
        author="Donglai Wei",
        python_requires=">=3.10",
        install_requires=[
            "numpy",
            "imageio",
        ],
        extras_require={
            "h5": ["h5py"],
            "zarr": ["zarr"],
            "cloud": ["cloud-volume"],
            "skel": ["kimimaro"],
            "all": ["h5py", "zarr", "cloud-volume", "kimimaro"],
        },
        packages=find_packages(),
    )


if __name__ == "__main__":
    # pip install --editable .
    setup_package()
