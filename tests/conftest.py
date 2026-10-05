import pytest
import torch

from rgc.determinism import default_device, set_determinism

set_determinism(0)

TEXT = (
    "The Normans were the people who in the 10th and 11th centuries gave their name to "
    "Normandy, a region in France. They were descended from Norse raiders and pirates."
)


@pytest.fixture(scope="session")
def device():
    return default_device()


@pytest.fixture(scope="session")
def model(device):
    from rgc.models import Variant, load

    return load(Variant(143000), device=device)


@pytest.fixture(scope="session")
def tokens(model):
    from rgc.models import tokenize

    return tokenize(model, TEXT)
