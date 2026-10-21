import pytest
import torch

from selfsne.neighbors import NearestNeighborSampler, knn_sampler


def similarity(x, y):
    return -(x[:, None] - y[None, :]).square().sum(-1)


def test_sampling_never_selects_excluded_entries_when_k_exceeds_valid_count():
    x = torch.tensor([[0.0], [1.0], [1.0]])
    for _ in range(20):
        indices = knn_sampler(x, x, similarity, num_neighbors=3)
        assert indices[0] in (1, 2)
        assert indices[1] == 0
        assert indices[2] == 0


@pytest.mark.parametrize('x', [torch.ones(1, 1), torch.ones(3, 1)])
def test_no_eligible_neighbors_is_an_explicit_error(x):
    with pytest.raises(ValueError, match='eligible'):
        knn_sampler(x, x, similarity, num_neighbors=2)


@pytest.mark.parametrize('k', [0, -1])
def test_nonpositive_neighbor_count_is_rejected(k):
    with pytest.raises(ValueError, match='num_neighbors'):
        NearestNeighborSampler(1, num_neighbors=k)


@pytest.mark.parametrize('column', [False, True])
def test_indexed_batches_preserve_index_shape_and_match_selected_data(column):
    sampler = NearestNeighborSampler(1, return_index=True, queue_size=8)
    data = torch.tensor([[0.0], [1.0], [4.0]])
    indices = torch.tensor([10, 20, 30])
    if column:
        indices = indices[:, None]
    result = sampler((data, indices))
    expected = torch.tensor([20, 10, 20])
    if column:
        expected = expected[:, None]
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize('evaluation', [False, True])
def test_frozen_or_evaluation_queue_does_not_mask_unrelated_diagonal(evaluation):
    sampler = NearestNeighborSampler(1, queue_size=2, freeze_queue_on_full=True)
    sampler(torch.tensor([[0.0], [10.0]]))
    if evaluation:
        sampler.eval()
    result = sampler(torch.tensor([[1.0], [9.0]]))
    torch.testing.assert_close(result, torch.tensor([[0.0], [10.0]]))


def test_empty_evaluation_queue_reports_missing_neighbors():
    sampler = NearestNeighborSampler(1).eval()
    with pytest.raises(ValueError, match='eligible'):
        sampler(torch.ones(2, 1))


def test_invalid_indices_do_not_desynchronize_queues():
    sampler = NearestNeighborSampler(1, return_index=True)
    with pytest.raises(ValueError, match='index'):
        sampler((torch.tensor([[0.0], [1.0]]), torch.tensor([1])))
    assert sampler.data_queue.queue_size == sampler.index_queue.queue_size == 0


def test_failed_selection_keeps_data_and_index_queues_aligned():
    sampler = NearestNeighborSampler(1, return_index=True, queue_size=4)
    with pytest.raises(ValueError, match='eligible'):
        sampler((torch.tensor([[0.0]]), torch.tensor([10])))
    assert sampler.data_queue.queue_size == sampler.index_queue.queue_size == 1
    result = sampler((torch.tensor([[2.0]]), torch.tensor([20])))
    torch.testing.assert_close(result, torch.tensor([10]))


def test_independent_queries_keep_diagonal_candidates_and_sample_only_top_k():
    x = torch.tensor([[0.0], [10.0]])
    y = torch.tensor([[1.0], [9.0], [30.0]])
    for _ in range(20):
        indices = knn_sampler(x, y, similarity, 2, exclude_diagonal=False)
        assert set(indices.tolist()) <= {0, 1}
    torch.testing.assert_close(
        knn_sampler(x, y, similarity, 1, exclude_diagonal=False), torch.tensor([0, 1])
    )


def test_invalid_similarities_are_excluded_without_mutating_kernel_output():
    scores = torch.tensor([[float('nan'), -2.0, float('inf')], [-1.0, -3.0, -4.0]])
    before = scores.clone()
    for _ in range(20):
        result = knn_sampler(torch.ones(2, 1), torch.ones(3, 1), lambda x, y: scores,
                             3, exclude_diagonal=False)
        assert result[0] == 1
    torch.testing.assert_close(scores, before, equal_nan=True)


def test_indexed_frozen_queue_checkpoint_preserves_selection():
    sampler = NearestNeighborSampler(1, queue_size=2, return_index=True,
                                     freeze_queue_on_full=True)
    sampler((torch.tensor([[0.0], [10.0]]), torch.tensor([11, 22])))
    restored = NearestNeighborSampler(1, queue_size=2, return_index=True,
                                      freeze_queue_on_full=True)
    restored.load_state_dict(sampler.state_dict())
    query = (torch.tensor([[1.0], [9.0]]), torch.tensor([33, 44]))
    torch.testing.assert_close(restored(query), torch.tensor([11, 22]))


@pytest.mark.parametrize('k', [0, -1])
def test_direct_sampler_rejects_nonpositive_neighbor_count(k):
    with pytest.raises(ValueError, match='num_neighbors'):
        knn_sampler(torch.ones(2, 1), torch.ones(2, 1), similarity, k)


def test_indexed_dataset_dataloader_batch_is_supported():
    from torch.utils.data import DataLoader
    from selfsne.neighbors import IndexedDataset

    data = torch.tensor([[0.0], [1.0], [4.0]])
    batch = next(iter(DataLoader(IndexedDataset(data), batch_size=3)))
    sampler = NearestNeighborSampler(1, return_index=True, queue_size=4)
    torch.testing.assert_close(sampler(batch), torch.tensor([1, 0, 1]))
