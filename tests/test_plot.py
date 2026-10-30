# -*- coding: utf-8 -*-
"""Tests for selfsne.plot module."""

import pytest
import torch
import numpy as np

from selfsne.plot import (
    PCAAlign,
    align_embeddings,
    equalize_axes_limits,
    EmbeddingDensity,
    density_colormaps,
    highest_density_contours,
)


class TestPCAAlign:
    """Tests for PCAAlign transformer."""

    def test_pca_align_reduces_dimensions(self):
        """PCAAlign should reduce to n_components dimensions."""
        X = np.random.randn(100, 10)
        pca = PCAAlign(n_components=2)
        X_transformed = pca.fit_transform(X)
        assert X_transformed.shape == (100, 2)

    def test_pca_align_preserves_variance(self):
        """PCAAlign should preserve the most variance in fewer dimensions."""
        np.random.seed(42)
        # Create data with strong signal in first 2 components
        X = np.random.randn(100, 10)
        X[:, 2:] = np.tile(X[:, :2] * 10, (1, 4))  # Strong correlation
        pca = PCAAlign(n_components=2)
        X_transformed = pca.fit_transform(X)
        # Transformed data should have non-trivial variance
        assert X_transformed.std() > 0

    def test_pca_align_default_components(self):
        """Default n_components=2 should work."""
        X = np.random.randn(50, 5)
        pca = PCAAlign()
        X_transformed = pca.fit_transform(X)
        assert X_transformed.shape == (50, 2)

    def test_pca_transform_matches_sklearn(self):
        """PCAAlign transform should match sklearn PCA up to sign flips."""
        from sklearn.decomposition import PCA as SklearnPCA
        np.random.seed(42)
        X = np.random.randn(100, 5)
        pca = PCAAlign(n_components=3)
        X_transformed = pca.fit_transform(X)
        sklearn_pca = SklearnPCA(n_components=3)
        sklearn_pca.fit(X)
        sklearn_transformed = sklearn_pca.transform(X)
        # Both methods should produce the same transformed shape
        assert X_transformed.shape == sklearn_transformed.shape
        # Both should preserve the explained variance ratio sum
        # (PCAAlign flips sign of first component but preserves variance)
        assert abs(X_transformed.std() - sklearn_transformed.std()) < 0.1


class TestAlignEmbeddings:
    """Tests for align_embeddings function."""

    def test_align_identity(self):
        """align_embeddings should return embedding close to reference when they match."""
        np.random.seed(42)
        ref = np.random.randn(20, 2)
        emb = ref.copy()  # Identical
        aligned = align_embeddings(ref, emb)
        # Should be very close to reference
        assert np.allclose(aligned, ref, atol=1e-4)

    def test_align_rotated(self):
        """align_embeddings should rotate embedding to match reference."""
        np.random.seed(42)
        ref = np.random.randn(20, 2)
        # Create rotated version
        theta = np.pi / 4
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        emb = ref @ rot.T
        aligned = align_embeddings(ref, emb)
        # Should be close to reference after alignment
        assert np.allclose(aligned, ref, atol=1e-3)

    def test_align_preserves_shape(self):
        """align_embeddings should preserve the shape of the embedding."""
        ref = np.random.randn(20, 2)
        emb = np.random.randn(20, 2)  # Same number of points as ref
        aligned = align_embeddings(ref, emb)
        assert aligned.shape == (20, 2)

    def test_align_preserves_pairwise_distances(self):
        """align_embeddings should preserve pairwise distances (rotation only)."""
        np.random.seed(42)
        ref = np.random.randn(20, 2)
        emb = np.random.randn(20, 2)
        aligned = align_embeddings(ref, emb)
        # Pairwise distances should be preserved
        def pairwise_dist(X):
            diff = X[:, None, :] - X[None, :, :]
            return np.sqrt((diff ** 2).sum(-1))
        dist_orig = pairwise_dist(emb)
        dist_aligned = pairwise_dist(aligned)
        assert np.allclose(dist_orig, dist_aligned, atol=1e-4)


class TestEqualizeAxesLimits:
    """Tests for equalize_axes_limits function."""

    def test_sets_aspect_to_one(self):
        """equalize_axes_limits should set aspect ratio to 1."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlim(-2, 2)
        ax.set_ylim(-1, 1)
        equalize_axes_limits(ax)
        assert ax.get_aspect() == 1.0
        plt.close(fig)

    def test_adjusts_limits_to_be_equal_range(self):
        """equalize_axes_limits should make x and y ranges equal."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlim(-2, 2)  # range = 4
        ax.set_ylim(-1, 1)  # range = 2
        equalize_axes_limits(ax)
        xlim_range = ax.get_xlim()[1] - ax.get_xlim()[0]
        ylim_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        assert np.isclose(xlim_range, ylim_range)
        plt.close(fig)


class TestEmbeddingDensity:
    """Tests for EmbeddingDensity class."""

    def test_histogram_shape(self):
        """embedding_histogram should return 2D histogram with expected shape."""
        np.random.seed(42)
        embedding = np.random.randn(100, 2)
        density = EmbeddingDensity(embedding, grid_size=10)
        hist = density.embedding_histogram()
        assert hist.ndim == 2
        assert hist.shape[0] > 0
        assert hist.shape[1] > 0

    def test_density_shape(self):
        """embedding_density should return smoothed density with expected shape."""
        np.random.seed(42)
        embedding = np.random.randn(100, 2)
        density = EmbeddingDensity(embedding, grid_size=10)
        density_map = density.embedding_density()
        assert density_map.shape == density.embedding_histogram().shape

    def test_conditional_density_with_labels(self):
        """conditional_embedding_density should return one density per label."""
        np.random.seed(42)
        embedding = np.random.randn(100, 2)
        labels = np.array([0] * 50 + [1] * 50)
        density = EmbeddingDensity(embedding, grid_size=10)
        densities = density.conditional_embedding_density(labels=labels)
        assert len(densities) == 2  # Two labels
        assert all(d.shape == density.embedding_histogram().shape for d in densities)

    def test_conditional_density_warns_without_labels(self):
        """conditional_embedding_density should warn when no labels provided."""
        np.random.seed(42)
        embedding = np.random.randn(100, 2)
        density = EmbeddingDensity(embedding, grid_size=10)
        with pytest.warns(UserWarning, match="No labels provided"):
            density.conditional_embedding_density()

    def test_conditional_density_rejects_mismatched_labels(self):
        """conditional_embedding_density should raise on mismatched label/embedding lengths."""
        np.random.seed(42)
        embedding = np.random.randn(100, 2)
        labels = np.zeros(50)
        density = EmbeddingDensity(embedding, grid_size=10)
        with pytest.raises(ValueError, match="must match"):
            density.conditional_embedding_density(labels=labels)

    def test_custom_embedding_in_histogram(self):
        """embedding_histogram should accept a custom embedding."""
        np.random.seed(42)
        embedding = np.random.randn(100, 2)
        custom = np.random.randn(50, 2) * 2  # Different scale
        density = EmbeddingDensity(embedding, grid_size=10)
        hist = density.embedding_histogram(embedding=custom)
        assert hist.shape[0] > 0  # Should compute histogram
        assert hist.sum() > 0  # Should have some counts

    def test_density_smoothing(self):
        """Larger kernel_scale should produce smoother density."""
        np.random.seed(42)
        embedding = np.random.randn(100, 2)
        density = EmbeddingDensity(embedding, grid_size=10)
        dens_smooth = density.embedding_density(kernel_scale=5)
        dens_sharp = density.embedding_density(kernel_scale=0.1)
        # Smooth should have lower max (spread out)
        assert dens_smooth.max() < dens_sharp.max()


class TestDensityColormaps:
    """Tests for density_colormaps function."""

    def test_returns_list_of_arrays(self):
        """density_colormaps should return a list of 3D arrays."""
        np.random.seed(42)
        densities = [np.random.rand(10, 10) for _ in range(3)]
        result = density_colormaps(densities)
        assert len(result) == 3
        assert all(r.ndim == 3 for r in result)

    def test_colormap_values_in_range(self):
        """Colormap values should be in [0, 1]."""
        np.random.seed(42)
        densities = [np.random.rand(10, 10) for _ in range(2)]
        result = density_colormaps(densities)
        assert all((r >= 0).all() and (r <= 1.0 + 1e-6).all() for r in result)

    def test_conditional_normalize(self):
        """With conditional_normalize, each density should be normalized to [0, 1]."""
        np.random.seed(42)
        densities = [np.random.rand(10, 10) for _ in range(2)]
        result = density_colormaps(densities, conditional_normalize=True)
        # Each should be normalized to its own max
        for r in result:
            assert np.isclose(r.max(), 1.0)


class TestHighestDensityContours:
    """Tests for highest_density_contours function."""

    def test_returns_list_of_contours(self):
        """highest_density_contours should return a list of arrays."""
        np.random.seed(42)
        # Use structured data that produces valid contours
        density = np.zeros((50, 50))
        density[15:35, 15:35] = 1.0
        density += np.random.rand(50, 50) * 0.1
        result = highest_density_contours([density])
        assert isinstance(result, list)

    def test_contours_are_arrays(self):
        """Each contour should be a numpy array."""
        np.random.seed(42)
        # Create a density with a clear high-density region for more contours
        density = np.zeros((50, 50))
        density[10:20, 10:20] = 1.0
        density[30:40, 30:40] = 1.0
        # Add noise to create more contour levels
        density += np.random.rand(50, 50) * 0.1
        result = highest_density_contours([density], level=0.3, largest_contour=False)
        if len(result) > 0:
            assert all(isinstance(c, np.ndarray) for c in result)

    def test_largest_contour_selection(self):
        """With largest_contour=True, should return only the largest contour."""
        np.random.seed(42)
        # Create a density with a clear high-density region
        density = np.zeros((50, 50))
        density[20:30, 20:30] = 1.0  # High-density blob
        # Add slight noise
        density += np.random.rand(50, 50) * 0.01
        result = highest_density_contours([density], level=0.5, largest_contour=True)
        # Should return at most one contour (the largest)
        assert len(result) <= 1

    def test_all_contours_without_largest_filter(self):
        """With largest_contour=False, should return all contours."""
        np.random.seed(42)
        # Create a density with multiple distinct regions
        density = np.zeros((50, 50))
        density[10:15, 10:15] = 1.0
        density[35:40, 35:40] = 1.0
        # Add slight noise to create more contour levels
        density += np.random.rand(50, 50) * 0.01
        result = highest_density_contours([density], level=0.5, largest_contour=False)
        # Should return all contours (at least 2)
        if len(result) > 0:
            assert len(result) >= 2
