import pytest
from src.evaluate import metrics, latency, majority_vote


def test_known_confusion():
    actual = metrics([1, 1, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0])
    assert {k: actual[k] for k in ('TP','TN','FP','FN')} == dict(TP=2,TN=2,FP=1,FN=1)
    for k in ('accuracy', 'precision', 'recall', 'F1'):
        assert actual[k] == pytest.approx(2/3)
    assert actual['FPR'] == pytest.approx(1/3)


def test_no_predictions_positive():
    m = metrics([0, 1], [0, 0])
    assert m['precision'] == m['recall'] == m['F1'] == m['FPR'] == 0


def test_latency_percentiles_and_units():
    m = latency([1, 2, 3, 4, 5])
    assert m['e2e_p50_ms'] == 3
    assert m['e2e_p95_ms'] == pytest.approx(4.8)
    assert m['e2e_throughput_images_s'] == pytest.approx(1000/3)


def test_vote_and_missing_model_rejected():
    assert majority_vote([[1,0,1], [1,0,0], [0,1,1]]) == [1,0,1]
    with pytest.raises(ValueError):
        majority_vote([[0], [1]])


def test_invalid_input_rejected():
    with pytest.raises(ValueError): metrics([], [])
    with pytest.raises(ValueError): metrics([0], [1,0])
    with pytest.raises(ValueError): latency([float('nan')])
