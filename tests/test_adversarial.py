
import sys
import os  # noqa
sys.path.insert(0, ".")  # noqa

import torch

torch.manual_seed(0)

from utils.styled_plot import plt
from utils.dataset import (
    load_test_image,
    preprocess_image,
    normalize_image,
    unnormalize_image,
    convert_idx_to_label
)
from classifiers.cnn_classifier import ImageNetClassifier
from tests.config import WORKING_DIR

import matplotlib

matplotlib.use('agg')

module = __import__(f"{WORKING_DIR}.adversarial", fromlist=[
    'get_gradient', 'perturb_image', 'create_adversarial', 'plot_adversarials'])

image = load_test_image()
image_preprocessed = preprocess_image(image)
image_preprocessed_norm = normalize_image(image_preprocessed).unsqueeze(0)

model = ImageNetClassifier()

y_pred, y_prob = model.predict(image_preprocessed_norm, return_probs=True)
print(f'Predicted class: "{convert_idx_to_label(y_pred.item())}". Confidence: {y_prob.item() * 100:.2f}%')
assert y_pred == torch.tensor([13])
assert torch.allclose(y_prob, torch.tensor([0.9483]), atol=1e-4)


def test_get_gradient():
    grad = module.get_gradient(model, torch.ones(1, 3, 224, 224))
    assert grad.shape == torch.Size([1, 3, 224, 224])
    assert torch.allclose(grad[0,0,0,0], torch.tensor([0.0056]), atol=1e-3)
    assert torch.allclose(grad[0,0,0,-1], torch.tensor([0.0075]), atol=1e-3)
    assert torch.allclose(grad.sum(), torch.tensor([0.2536]), atol=5e-2)

    grad = module.get_gradient(model, image_preprocessed_norm)
    assert grad.shape == torch.Size([1, 3, 224, 224])
    assert torch.allclose(grad[0,0,0,0], torch.tensor([0.0005]), atol=1e-3)
    assert torch.allclose(grad[0,0,0,-1], torch.tensor([0.0]), atol=1e-3)
    assert torch.allclose(grad.abs().sum(), torch.tensor([59.4565]), atol=2.0)



def test_perturb_image():
    image = module.perturb_image(image_preprocessed_norm, image_preprocessed_norm, 0.1)
    assert image.shape == torch.Size([1, 3, 224, 224])
    assert torch.allclose(image[0,0,0,0], torch.tensor([0.4823]), atol=1e-2)
    assert torch.allclose(image[0,0,0,-1], torch.tensor([-0.8137]), atol=1e-2)
    assert torch.allclose(image.abs().sum(), torch.tensor([112889.46]), atol=100.0)

    image = module.perturb_image(image_preprocessed_norm, image_preprocessed, 10.0)
    assert torch.allclose(image[0,0,0,0], torch.tensor([10.3823]), atol=1e-1)
    assert torch.allclose(image[0,0,0,-1], torch.tensor([9.2863]), atol=1e-1)
    assert torch.allclose(image.abs().sum(), torch.tensor([1518113.0]), atol=100.0)

    image = module.perturb_image(image_preprocessed_norm, image_preprocessed, 0.0)
    assert torch.allclose(image[0,0,0,0], image_preprocessed_norm[0,0,0,0], atol=1e-1)
    assert torch.allclose(image[0,0,0,-1], image_preprocessed_norm[0,0,0,-1], atol=1e-1)
    assert torch.allclose(image.abs().sum(), image_preprocessed_norm.abs().sum(), atol=100.0)


def test_create_adversarial():
    advs = module.create_adversarials(model, image_preprocessed_norm, [0.6])
    assert isinstance(advs, list)
    assert len(advs) == 1
    assert advs[0].shape == torch.Size([1, 3, 224, 224])
    assert model.predict(advs[0]) != torch.tensor([13])

    advs = module.create_adversarials(model, image_preprocessed_norm, [0.6, 0.8, 0.0, 0.01])
    assert isinstance(advs, list)
    assert len(advs) == 4
    for adv in advs:
        assert adv.shape == torch.Size([1, 3, 224, 224])
    assert model.predict(advs[0]) != torch.tensor([13])
    assert model.predict(advs[1]) != torch.tensor([13])
    assert model.predict(advs[2]) == torch.tensor([13])
    assert model.predict(advs[3]) == torch.tensor([12])


def test_plot_adversarials():
    advs = [image_preprocessed_norm * 0, image_preprocessed_norm * 2]
    eps_vals = [0.1, 0.2]
    preds = ['letter opener', 'house finch']
    module.plot_adversarials(model, image_preprocessed_norm, advs, eps_vals)

    fig = plt.gcf()

    # test if correct number of subplots
    assert len(fig.axes) == 1 + len(advs)

    # test if image is plotted correctly:
    aximg = [x for x in fig.axes[0].get_children() if isinstance(x, matplotlib.image.AxesImage)][0]
    assert torch.allclose(torch.tensor(aximg.get_array()).permute(2, 0, 1)[0], image_preprocessed[0])

    # test if last adv is plotted correctly:
    aximg = [x for x in fig.axes[-1].get_children() if isinstance(x, matplotlib.image.AxesImage)][0]
    assert torch.allclose(torch.tensor(aximg.get_array()).permute(2, 0, 1), unnormalize_image(advs[-1])[0])

    # test if eps values and predictions are used in the titles of subplots
    assert 'junco' in fig.axes[0].get_title()
    for i, ax in enumerate(fig.axes[1:]):
        assert str(eps_vals[i]) in ax.get_title()
        assert str(preds[i]) in ax.get_title()


if __name__ == "__main__":
    test_get_gradient()
    test_perturb_image()
    test_create_adversarial()
    test_plot_adversarials()
