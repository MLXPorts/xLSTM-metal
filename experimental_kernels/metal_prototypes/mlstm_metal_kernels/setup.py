
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
from setuptools import setup, Extension
import pkg_resources

# Custom build class to handle Metal files
class MetalBuildExt(build_ext):
    def build_extensions(self):
        # Add Metal framework linking
        for ext in self.extensions:
            ext.extra_link_args += ['-framework', 'Metal', '-framework', 'Foundation']
        super().build_extensions()

ext_modules = [
    Pybind11Extension(
        "mlstm_metal_backend",
        ["mlstm_metal_backend.mm"],
        include_dirs=[pybind11.get_include()],
        language='c++',
        cxx_std=17,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": MetalBuildExt},
    zip_safe=False,
)
