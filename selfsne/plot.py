# -*- coding: utf-8 -*-
# Copyright 2021 Jacob M. Graving <jgraving@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.measure import find_contours
import shapely.geometry as geometry

from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin


class PCAAlign(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, copy=True):
        self.pca = PCA(n_components, copy=copy)

    def fit(self, X, y=None):
        self.pca.fit(X)
        eigenvectors = self.pca.components_
        signs = np.sign(eigenvectors[0])
        self.pca.components_ = eigenvectors * signs
        return self

    def transform(self, X):
        return self.pca.transform(X)


def align_embeddings(reference_embedding, embedding):
    def cost_function(params):
        theta = params[0]
        rotation_matrix = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        rotated_embedding = np.matmul(embedding, rotation_matrix)
        return np.linalg.norm(
            rotated_embedding / np.linalg.norm(rotated_embedding)
            - reference_embedding / np.linalg.norm(reference_embedding)
        )

    initial_guess = [0]
    result = minimize(cost_function, initial_guess, method="BFGS")
    theta = result.x[0]
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    rotated_embedding = np.matmul(embedding, rotation_matrix)
    return rotated_embedding


class EmbeddingDensity:
    def __init__(
        self, embedding, limit_margin=0.1, limit_percentile=100, grid_size=100
    ):
        self.embedding = np.array(embedding)
        percentile_offset = (100 - limit_percentile) / 2
        percentile_range = [0 + percentile_offset, 100 - percentile_offset]
        embedding_range = np.percentile(self.embedding, percentile_range, axis=0)
        self.x_min, self.x_max = embedding_range[0, 0], embedding_range[1, 0]
        self.y_min, self.y_max = embedding_range[0, 1], embedding_range[1, 1]

        self.x_range = self.x_max - self.x_min
        self.y_range = self.y_max - self.y_min

        self.x_min_adj = self.x_min - self.x_range * limit_margin
        self.x_max_adj = self.x_max + self.x_range * limit_margin
        self.y_min_adj = self.y_min - self.y_range * limit_margin
        self.y_max_adj = self.y_max + self.y_range * limit_margin

        aspect_ratio = self.x_range / self.y_range
        self.grid_size_x = int(grid_size * aspect_ratio)
        self.grid_size_y = grid_size

        self.x_grid = np.linspace(self.x_min_adj, self.x_max_adj, self.grid_size_x)
        self.y_grid = np.linspace(self.y_min_adj, self.y_max_adj, self.grid_size_y)

    def embedding_histogram(self, embedding=None):
        if embedding is None:
            embedding = self.embedding
        else:
            embedding = np.array(embedding)

        # Create a 2D histogram of the embedding using the x_grid and y_grid
        histogram, _, _ = np.histogram2d(
            embedding[:, 0],
            embedding[:, 1],
            bins=[self.x_grid, self.y_grid],
            density=True,
        )
        return histogram

    def embedding_density(self, embedding=None, kernel_scale=1):
        histogram = self.embedding_histogram(embedding)
        density = gaussian_filter(
            histogram, sigma=kernel_scale, mode="constant", cval=0
        )
        return density

    def conditional_embedding_density(
        self, embedding=None, labels=None, kernel_scale=1
    ):
        if embedding is None:
            embedding = self.embedding
        else:
            embedding = np.array(embedding)

        # If labels is None, generate a single label for all the embedding vectors
        if labels is None:
            warnings.warn(
                "No labels provided, generating a single density for all the embedding vectors"
            )
            labels = np.zeros(len(embedding))

        # Make sure the first axis of labels and embedding match
        if labels is not None and len(labels) != len(embedding):
            raise ValueError("The first axis of labels and embedding must match")

        # Get the unique labels
        unique_labels = np.unique(labels)

        # Initialize the list of conditional densities
        conditional_densities = []

        # Iterate over the unique labels
        for label in unique_labels:
            # Mask the embedding to select only the vectors with the current label
            mask = labels == label
            masked_embedding = embedding[mask]

            # Calculate the density for the masked embedding
            conditional_density = self.embedding_density(
                embedding=masked_embedding, kernel_scale=kernel_scale
            )

            # Add the conditional density to the list
            conditional_densities.append(conditional_density)

        return conditional_densities


def density_colormaps(
    embedding_densities, conditional_normalize=True, colormap="tab10"
):
    # Get the colormap
    cmap = plt.get_cmap(colormap)

    # Initialize the list of density colormaps
    density_colormaps = []

    # Iterate over the embedding densities
    for idx, embedding_density in enumerate(embedding_densities):
        # Normalize the embedding density
        if conditional_normalize:
            normalized_density = embedding_density / embedding_density.max()
        else:
            normalized_density = embedding_density / np.max(embedding_densities)

        # Get the color for the embedding density
        color = cmap(idx)[:4]

        # Convert the embedding density to a 4-channel color array by multiplying it by the color
        density_colormap = normalized_density[:, :, np.newaxis] * color

        # Add the density colormap to the list
        density_colormaps.append(density_colormap)

    return density_colormaps


def highest_density_contours(embedding_densities, level=0.9, largest_contour=True):
    # Initialize the list of highest density contours
    highest_density_contours = []

    # Iterate over the embedding densities
    for embedding_density in embedding_densities:
        # Calculate the highest density contour for the embedding density
        contours = find_contours(embedding_density, level=level)

        # Select the largest contour by area, if requested
        if largest_contour and len(contours) > 1:
            areas = [geometry.Polygon(contour).area for contour in contours]
            largest_contour_index = np.argmax(areas)
            contours = [contours[largest_contour_index]]

        # Add the selected contours to the list
        highest_density_contours.extend(contours)

    return highest_density_contours
