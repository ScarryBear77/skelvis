import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='skelvis',
    version='0.100',
    scripts=[],
    author="Peter Kovacs",
    author_email="petersmith77.sb@gmail.com",
    description="3D skeleton visualizer package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ScarryBear77/skelvis",
    packages=setuptools.find_packages(),
    install_requires=['k3d'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
 )
