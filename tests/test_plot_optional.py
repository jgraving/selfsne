import numpy as np
import pytest

try:
    import matplotlib
    has_matplotlib = True
except ImportError:
    has_matplotlib = False

try:
    import sklearn
    has_sklearn = True
except ImportError:
    has_sklearn = False

has_deps = has_matplotlib and has_sklearn
pytestmark = pytest.mark.skipif(not has_deps, reason="matplotlib or scikit-learn not installed")

# Guard import of selfsne.plot which requires optional deps
if has_deps:
    from selfsne.plot import EmbeddingDensity, PCAAlign, align_embeddings


def test_embedding_density_histogram_shape():
    embedding = np.random.randn(100, 2)
    ed = EmbeddingDensity(embedding)
    hist = ed.embedding_histogram()
    # histogram2d returns shape (len(x_grid)-1, len(y_grid)-1)
    assert hist.shape == (ed.grid_size_x - 1, ed.grid_size_y - 1)


def test_embedding_density_conditional():
    embedding = np.random.randn(200, 2)
    labels = np.array([0]*100 + [1]*100)
    ed = EmbeddingDensity(embedding)
    cond = ed.conditional_embedding_density(labels=labels)
    assert len(cond) == 2
    for d in cond:
        assert d.shape == (ed.grid_size_x - 1, ed.grid_size_y - 1)


def test_pca_align_fit_transform():
    X = np.random.randn(100, 2)
    pca = PCAAlign()
    pca.fit(X)
    Xt = pca.transform(X)
    assert Xt.shape == (100, 2)


def test_align_embeddings_shape():
    ref = np.random.randn(100, 2)
    emb = np.random.randn(100, 2)
    rotated = align_embeddings(ref, emb)
    assert rotated.shape == emb.shape
