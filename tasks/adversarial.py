
import sys
import os  # noqa
sys.path.insert(0, ".")  # noqa

import torch
from utils.styled_plot import plt
from utils.dataset import (
    load_test_image,
    preprocess_image,
    normalize_image,
    unnormalize_image,
    convert_idx_to_label
)
from classifiers.cnn_classifier import ImageNetClassifier

torch.manual_seed(0)


def get_gradient(model, image):
    """
    Propagates the loss between the model's output and the predicted label back to the input to get the input gradient.

    Parameters:
        model (ImageNetClassifier, torch.nn.Module):
            The image classification model. This is a torch.nn.Module, so you can call its forward method using `model()`.
            Also has a `.predict` method that returns the index of the predicted label.

        image (torch.tensor): The input for which to compute the gradient.

    Returns:
        gradient (torch.tensor): The input gradient. Same shape as the input image.
    """

    return None


def perturb_image(image, grad, eps):
    """
    Applies a perturbation to an image based on the Fast-Gradient_sign method.

    Parameters:
        image (torch.tensor): The image to perturb.

        grad (torch.tensor): The input gradient corresponding to the image.

        eps (float): The epsilon value for the perturbation, specifying the magnitude of the perturbation.

    Returns:
        image (torch.tensor): The perturbed image.
    """

    return None


def create_adversarials(model, image, eps_values):
    """
    Creates adversarial examples for the given image and model using the Fast-Gradient_sign method.
    (Refer to W08 T02 slide 12).

    Parameters:
        model (ImageNetClassifier, torch.nn.Module):
            The image classification model. This is a torch.nn.Module, so you can call its forward method using `model()`.
            Also has a `.predict` method that returns the index of the predicted label.

        image (torch.tensor): The image to generate adversarial examples from.

        eps_values (List[float]): The list of epsilon values for which to generate adversarial examples.

    Returns:
        adversarials (List[torch.tensor]): A list containing one adversarial example for each eps value in eps_values.
    """

    return None



def plot_adversarials(model, image, adv_images, eps_values):
    """
    Plots the  original image and the adversarial images in a single row.
    Uses the eps value and the predicted label as axis titles.

    Parameters:
        model (ImageNetClassifier, torch.nn.Module):
            The image classification model. This is a torch.nn.Module, so you can call its forward method using `model()`.
            Also has a `.predict` method that returns the index of the predicted label.

        image (torch.tensor): The original image corresponding to the adversarial examples.

        adv_images (List[torch.tensor]): A list containing the adversarial examples to visualize.

        eps_values (List[float]): The list of epsilon values corresponding to each adversarial example in adv_images.

    Hint: 
        - you can use convert_index_to_label to convert the predicted class indices to class labels.
        - matplotlib expects a channels last format
        - The model works with normalized images. Before visualizing the images, you have to invert the normalization using `unnormalize()`
    """

    fig, axes = plt.subplots(len(adv_images) + 1, 1)


if __name__ == "__main__":
    image = load_test_image()
    image_preprocessed = preprocess_image(image)
    image_preprocessed_norm = normalize_image(image_preprocessed).unsqueeze(0)

    model = ImageNetClassifier()
    
    y_pred, y_prob = model.predict(image_preprocessed_norm, return_probs=True)
    print(f'Predicted class: "{convert_idx_to_label(y_pred.item())}". Confidence: {y_prob.item() * 100:.2f}%')
    assert y_pred == torch.tensor([13])
    assert torch.allclose(y_prob, torch.tensor([0.9483]), atol=1e-4)

    eps_values = [0.0001, 0.001, 0.01, 0.1, 0.3]
    print('Running `create_adversarial` ...')
    adv_images = create_adversarials(model, image_preprocessed_norm, eps_values)

    print('Running `plot_adversarials` ...')
    plot_adversarials(model, image_preprocessed_norm, adv_images, eps_values)
    plt.show()
