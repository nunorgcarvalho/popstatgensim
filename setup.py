from setuptools import Extension, setup

import numpy as np


setup(
    ext_modules=[
        Extension(
            "popstatgensim._reml_accel",
            sources=["src/popstatgensim/_reml_accel.c"],
            include_dirs=[np.get_include()],
            extra_compile_args=["-O3"],
        )
    ]
)
