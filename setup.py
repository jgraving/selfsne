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
LONG_DESCRIPTION = """Self-SNE is a probabilistic self-supervised deep learning model for compressing high-dimensional data to a low-dimensional embedding. It is a general-purpose algorithm that works with multiple types of data including images, sequences, and tabular data. It uses self-supervised objectives, such as InfoNCE, to preserve structure in the compressed latent space. Self-SNE can also (optionally) simultaneously learn a cluster distribution (a prior over the latent embedding) during optimization. Overlapping clusters are automatically combined by optimizing a variational upper bound on entropy, so the number of clusters does not have to be specified manually — provided the number of initial clusters is large enough. Self-SNE produces embeddings with similar quality to existing dimensionality reduction methods; can detect outliers; scales to large, out-of-core datasets; and can easily add new data to an existing embedding/clustering.

"""

DISTNAME = "selfsne"
MAINTAINER = "Jacob Graving <jgraving@gmail.com>"
MAINTAINER_EMAIL = "jgraving@gmail.com"
URL = "https://github.com/jgraving/selfsne"
LICENSE = "Apache 2.0"
DOWNLOAD_URL = "https://github.com/jgraving/selfsne.git"
VERSION = "0.0.2.dev"


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
