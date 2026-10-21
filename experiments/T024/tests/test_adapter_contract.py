import pytest
from src.adapters.base import validate_prediction
from src.adapters.safe_checkpoint import restricted_load_type


def prediction(score, label):
    return dict(score=score,label=label,boxes=[],timing_ms=1.)


def test_threshold_boundary():
    assert validate_prediction(prediction(.5, 1))['label'] == 1
    assert validate_prediction(prediction(.499, 0))['label'] == 0


@pytest.mark.parametrize('score,label', [(float('nan'),0), (float('inf'),1), (-.1,0), (1.1,1), (.5,0)])
def test_invalid_probabilities_or_labels_rejected(score,label):
    with pytest.raises(ValueError): validate_prediction(prediction(score,label))


def test_dill_shim_cannot_resolve_arbitrary_types():
    assert restricted_load_type('set') is set
    with pytest.raises(ValueError): restricted_load_type('eval')
