from setuptools import setup, Extension
import numpy as np

extensions = [
    Extension(
        "DistMLIP.distributed.subgraph_creation_fast",
        sources=["DistMLIP/distributed/subgraph_creation_fast.c", "DistMLIP/distributed/subgraph_creation_utils.c", "DistMLIP/distributed/fpis.c"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        # define_macros=[("DEBUG", ""), ("TIMING", "")],
        define_macros=[("DEBUG", "")],
        include_dirs=[np.get_include()])
]

setup(ext_modules=extensions)
