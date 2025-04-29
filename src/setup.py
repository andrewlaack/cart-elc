
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "decision_tree",  # Name of the Python module
        sources=[
            "src/module.cpp",       # Pybind11 wrapper source
            "src/ELCClassifier.cpp",  # Your custom class implementation
            "src/TreeNode.cpp",     # Dependency source
        ],
        include_dirs=["include"],  # Directory for header files
        extra_compile_args=["-std=c++11", ] # "-03"],  # Enable C++11 standard
    ),
]

setup(
    name="decision_tree",
    version="0.1",
    description="Pybind11 wrapper for ELCClassifier",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
