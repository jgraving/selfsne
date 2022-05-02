# -*- coding: utf-8 -*-
"""
Copyright 2021 Jacob M. Graving <jgraving@gmail.com>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import sys
import warnings
from setuptools import setup, find_packages

DESCRIPTION = "Self-Supervised Noise Embeddings (Self-SNE) for dimensionality reduction and clustering"
LONG_DESCRIPTION = """
"""

DISTNAME = "selfsne"
MAINTAINER = "Jacob Graving <jgraving@gmail.com>"
MAINTAINER_EMAIL = "jgraving@gmail.com"
URL = "https://github.com/jgraving/selfsne"
LICENSE = "Apache 2.0"
DOWNLOAD_URL = "https://github.com/jgraving/selfsne.git"
VERSION = "0.0.dev"


if __name__ == "__main__":

    setup(
        name=DISTNAME,
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        license=LICENSE,
        url=URL,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        install_requires=["numpy", "tqdm"],
        packages=find_packages(),
        zip_safe=False,
        classifiers=[
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3",
            "Operating System :: Unix",
            "Operating System :: MacOS",
        ],
    )
