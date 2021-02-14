import numpy as np

from fastapi.testclient import TestClient
from ...main import app
from ...routers import datasets as ds

client = TestClient(app)


def test_moving_average_smoother():
    xs = np.array([1,1,1,1,1,2,2,2,2,2])
    mas = ds.MovingAverageSmoother(method='centered', width=5)
    smoothed_xs = mas.apply(xs)
    assert np.allclose(smoothed_xs, np.array([1, (4 + 2)/5, (3 + 4)/5, (2 + 6)/5, (1 + 8)/5, 2]))

