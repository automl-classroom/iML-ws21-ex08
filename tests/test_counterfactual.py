import sys
import os  # noqa
sys.path.insert(0, ".")  # noqa

import torch
torch.manual_seed(0)

from classifiers.mlp_classifier import MLPClassifier, WEIGHTS_FILE
from tests.config import WORKING_DIR
from utils.wheat_seeds_dataset import Dataset, PyTorchDataset


module = __import__(f"{WORKING_DIR}.counterfactual", fromlist=[
    'compute_distance', 'compute_output_difference', 'compute_loss', 'create_counterfactual'])

dataset = Dataset(
    "wheat_seeds",
    [0, 1, 2, 3, 4, 5, 6],
    [7],
    normalize=True,
    categorical=True)

(X_train, y_train), (X_test, y_test) = dataset.get_data()

input_units = X_train.shape[1]
output_units = len(dataset.get_classes())
features = dataset.get_input_labels()

ds_train = PyTorchDataset(X_train, y_train)
ds_test = PyTorchDataset(X_test, y_test)

model = MLPClassifier(input_units, output_units, 20)

if WEIGHTS_FILE.exists():
    model.load_state_dict(torch.load(WEIGHTS_FILE))
else:
    model.fit(ds_train, ds_test)
    torch.save(model.state_dict(), WEIGHTS_FILE)


def test_compute_distance():

    distance = module.compute_distance(torch.zeros(3), torch.zeros(3))
    assert torch.allclose(distance, torch.tensor(0.0))
    assert distance.shape == torch.Size([])

    distance = module.compute_distance(torch.ones(3), torch.zeros(3))
    assert torch.allclose(distance, torch.tensor(3.0))

    distance = module.compute_distance(torch.full((3,), 3.0), torch.zeros(3))
    assert torch.allclose(distance, torch.tensor(9.0))

    distance = module.compute_distance(torch.full((3,), -1.0), torch.full((3,), -4.0))
    assert torch.allclose(distance, torch.tensor(9.0))

    distance = module.compute_distance(torch.tensor([-123.0]), torch.tensor([99.0]))
    assert torch.allclose(distance, torch.tensor(222.0))

    
def test_compute_output_difference():

    difference = module.compute_output_difference(torch.zeros(3), torch.zeros(3))
    assert torch.allclose(difference, torch.tensor(0.0))
    assert difference.shape == torch.Size([])

    difference = module.compute_output_difference(torch.ones(3), torch.zeros(3))
    assert torch.allclose(difference, torch.tensor(3.0))

    difference = module.compute_output_difference(torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0.3, 0.2, 0.1]))
    assert torch.allclose(difference, torch.tensor(0.08))

    difference = module.compute_output_difference(torch.tensor([-123.0]), torch.tensor([99.0]))
    assert torch.allclose(difference, torch.tensor(49284.0))


def test_compute_loss():

    loss = module.compute_loss(torch.zeros(5), torch.zeros(5), torch.zeros(3), torch.zeros(3), 1)
    assert torch.allclose(loss, torch.tensor(0.0))
    assert loss.shape == torch.Size([])

    loss = module.compute_loss(torch.ones(5), torch.zeros(5), torch.ones(3), torch.zeros(3), 1)
    assert torch.allclose(loss, torch.tensor(8.0))

    loss = module.compute_loss(torch.ones(5), torch.zeros(5), torch.ones(3), torch.zeros(3), 0.5)
    assert torch.allclose(loss, torch.tensor(6.5))

    loss = module.compute_loss(torch.ones(5), torch.zeros(5), torch.ones(3), torch.zeros(3), 5)
    assert torch.allclose(loss, torch.tensor(20.0))

    loss = module.compute_loss(torch.tensor([0.1, 0.2]), torch.tensor([0.4, 0.1]), torch.tensor([0.1, 0.4, 0.5]), torch.tensor([0.5, 0.4, 0.1]), 0.7)
    assert torch.allclose(loss, torch.tensor(0.624))


def test_create_counterfactual():

    x = ds_test[0][0].unsqueeze(0)
    cf, step = module.create_counterfactual(model, x, 0, lambda_reg=1.0, num_steps=1000)

    assert isinstance(step, int)
    assert cf.shape == x.shape
    assert step <= 215
    assert torch.allclose(cf[0, 0], torch.tensor([0.3205]), atol=1e-2)
    assert torch.allclose(cf[0, -1], torch.tensor([0.1291]), atol=1e-2)

    prediction = model.predict(cf)
    assert prediction == torch.tensor([0])


    x = ds_test[1][0].unsqueeze(0)
    cf, step = module.create_counterfactual(model, x, 1, lambda_reg=5.0, num_steps=1000)

    assert isinstance(step, int)
    assert cf.shape == x.shape
    assert step <= 250
    assert torch.allclose(cf[0, 0], torch.tensor([0.6025]), atol=1e-2)
    assert torch.allclose(cf[0, -1], torch.tensor([0.5290]), atol=1e-2)

    prediction = model.predict(cf)
    assert prediction == torch.tensor([1])


if __name__ == "__main__":
    test_compute_distance()
    test_compute_output_difference()
    test_compute_loss()
    test_create_counterfactual()
