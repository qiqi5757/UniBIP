from setuptools import setup, find_packages

setup(
    name="UniBIP",
    version="0.1.0",
    author="Your Name or Team Name",
    description="A Python module for UniBIP modeling and analysis.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/unibip-module", # Replace with your GitHub URL
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        # List your dependencies here
        'torch',
        'torch_geometric',
        'torch_scatter',
        'scikit-learn',
        'numpy',
        'pandas',
        'scipy',
        'networkx',
        'joblib',
        'tqdm',
        'setproctitle',
        'pyarrow', # Used for feather files
    ],
    # Include data files in the package
    package_data={
        'unibip_module': ['dta/kiba/*.csv', 'dta/kiba/*.txt'],
    },
    include_package_data=True,
)