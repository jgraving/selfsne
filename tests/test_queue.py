import io

import pytest
import torch

from selfsne.neighbors import Queue


@pytest.mark.parametrize("count", [0, 2, 4, 7])
@pytest.mark.parametrize("freeze", [False, True])
def test_checkpoint_preserves_queue_occupancy_and_next_update(count, freeze):
    queue = Queue(2, queue_size=4, freeze_on_full=freeze)
    queue(torch.arange(count * 2, dtype=torch.float32).reshape(count, 2))
    checkpoint = io.BytesIO()
    torch.save(queue.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = Queue(2, queue_size=4, freeze_on_full=freeze)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    assert restored.queue_size == min(count, 4)
    assert restored.full == queue.full
    assert restored.freeze == queue.freeze
    update = torch.full((2, 2), -1.0)
    torch.testing.assert_close(restored(update), queue(update))


def test_eval_only_returns_valid_rows_and_does_not_update():
    queue = Queue(2, queue_size=4)
    data = torch.tensor([[1.0, 2.0]])
    queue(data)
    queue.eval()
    torch.testing.assert_close(queue(torch.zeros(3, 2)), data)
    assert queue.queue_size == 1


def test_empty_queue_evaluation_and_zero_samples():
    queue = Queue(2, queue_size=4).eval()
    assert queue(torch.ones(3, 2)).shape == (0, 2)
    assert queue.sample(0).shape == (0, 2)


def test_queue_keeps_recent_batches_and_detaches_inputs():
    queue = Queue(1, queue_size=4)
    queue(torch.tensor([[1.0], [2.0]], requires_grad=True))
    result = queue(torch.tensor([[3.0], [4.0], [5.0]], requires_grad=True))
    torch.testing.assert_close(result, torch.tensor([[3.0], [4.0], [5.0], [1.0]]))
    assert not result.requires_grad
    assert set(queue.sample(4).flatten().tolist()) <= {1.0, 3.0, 4.0, 5.0}


@pytest.mark.parametrize("features,capacity", [(0, 4), (2, 0), (-1, 4)])
def test_queue_rejects_invalid_dimensions(features, capacity):
    with pytest.raises(ValueError):
        Queue(features, queue_size=capacity)


@pytest.mark.parametrize("count", [-1, 1])
def test_empty_queue_rejects_invalid_sample_count(count):
    with pytest.raises(ValueError, match="n_samples"):
        Queue(2).sample(count)


def test_queue_rejects_wrong_feature_shape_without_mutating_state():
    queue = Queue(2, queue_size=4)
    with pytest.raises(ValueError, match="shape"):
        queue(torch.ones(3, 1))
    assert queue.queue_size == 0
    assert queue.queue.shape == (4, 2)


def test_queue_updates_preserve_previous_results_and_configured_dtype():
    queue = Queue(1, queue_size=3, dtype=torch.float32)
    previous = queue(torch.ones(1, 1, dtype=torch.float64))
    queue(torch.full((1, 1), 2.0))
    torch.testing.assert_close(previous, torch.ones(1, 1))
    assert queue.queue.dtype == torch.float32
