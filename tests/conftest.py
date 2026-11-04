import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import torch


@pytest.fixture(autouse=True, scope="session")
def deterministic_cpu():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
