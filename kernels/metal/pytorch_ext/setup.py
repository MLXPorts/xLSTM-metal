from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

class MetalBuildExt(build_ext):
    def build_extensions(self):
        for ext in self.extensions:
            ext.extra_link_args += ["-framework", "Metal", "-framework", "Foundation"]
        super().build_extensions()

ext_modules = [
    Pybind11Extension(
        "mlstm_metal_backend",
        ["mlstm_metal_backend.mm"],
        cxx_std=17,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": MetalBuildExt},
    zip_safe=False,
)

