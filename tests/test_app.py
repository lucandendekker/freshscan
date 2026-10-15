import io

import numpy as np
import pytest
from PIL import Image

from app import create_app, prepare_image


def image_upload():
    data = io.BytesIO()
    Image.new('RGB', (50, 30), (255, 128, 0)).save(data, format='PNG')
    data.seek(0)
    return data


class Predictor:
    def predict(self, batch, verbose=0):
        assert batch.shape == (1, 224, 224, 3)
        assert batch.dtype == np.float32
        assert 0 <= batch.min() <= batch.max() <= 1
        return np.array([[0.15, 0.85]])


@pytest.fixture
def client():
    return create_app(Predictor()).test_client()


def test_preprocessing_matches_training():
    batch = prepare_image(image_upload().getvalue())
    np.testing.assert_allclose(batch[0, 0, 0], [1, 128 / 255, 0])


def test_prediction_contract(client):
    response = client.post('/predict', data={'file': (image_upload(), 'sample.png')})
    assert response.status_code == 200
    assert response.json == {'class': 'rotten', 'fresh_confidence': 15.0, 'rotten_confidence': 85.0}


def test_invalid_uploads(client):
    assert client.post('/predict').status_code == 400
    assert client.post('/predict', data={'file': (io.BytesIO(b'not an image'), 'bad.png')}).status_code == 400
    assert client.post('/predict', data={'file': (io.BytesIO(b'x' * (11 * 1024 * 1024)), 'large.png')}).status_code == 413


def test_only_public_assets_are_served(client):
    assert client.get('/').status_code == 200
    assert client.get('/potato_imgs/fresh/fresh_0057.jpg').status_code == 200
    assert client.get('/cnn_v4_best.keras').status_code == 404
    assert client.get('/app.py').status_code == 404


def test_real_model_predictions():
    from app import BASE_DIR, load_model

    client = create_app(load_model()).test_client()
    for expected, filename in [('fresh', 'fresh_0057.jpg'), ('rotten', 'rotten_0016.jpg')]:
        with (BASE_DIR / 'potato_imgs' / expected / filename).open('rb') as image:
            response = client.post('/predict', data={'file': (image, filename)})
        assert response.status_code == 200
        assert response.json['class'] == expected
        assert abs(response.json['fresh_confidence'] + response.json['rotten_confidence'] - 100) < 0.2
