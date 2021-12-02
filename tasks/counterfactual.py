import sys
import os  # noqa
# os.chdir('..')
sys.path.insert(0, ".")  # noqa

from pathlib import Path
import pandas as pd
import torch
from utils.styled_plot import plt
from utils.wheat_seeds_dataset import Dataset, PyTorchDataset
from classifiers.mlp_classifier import MLPClassifier, WEIGHTS_FILE


torch.manual_seed(0)


def compute_distance(x, counterfactual):
    """
    Computes the distance between the original input and a counterfactual, using the L1 distance.
    
    Parameters:
        x (torch.tensor): The original input, of shape (1, num_features)
        counterfactual (torch.tensor): The counterfactual input. Same shape as x.

    Returns:
        distance (torch.tensor): The L1 distance between the original input x and the counterfactual input.
    """
    return None


def compute_output_difference(output, desired_output):
    """
    Computes the difference between the model's output and a desired target output based on the summed square error.
    
    Parameters:
        output (torch.tensor): The models output for a single instance of shape (1, 3).
        desired_output (torch.tensor): The target output, of same shape as the output.

    Returns:
        difference (torch.tensor): The summed square error between output and desired_output.
    """
    return None


def compute_loss(x, counterfactual, output, desired_output, lambda_reg):
    """
    Computes the loss that is minimized during counterfactual generation.
    The loss has two components: 
        - the distance between the original instance x and the counterfactual instance
        - the difference between the model output and the desired output.
    The two components are combined via a weighted sum (weighted by lambda_reg).
    Refer to W08 T03 slide 21.
    
    Parameters:
        x (torch.tensor): The original input, of shape (1, num_features)
        counterfactual (torch.tensor): The counterfactual input. Same shape as x.
        output (torch.tensor): The models output for a single instance of shape (1, 3).
        desired_output (torch.tensor): The target output, of same shape as the output.
        lambda_reg (float): The lambda denoting the regularization strength.

    Returns:
       loss (torch.tensor): The loss value, distance(x, x') + lambda_reg * difference(f(x'), y')
    """
    return None


def create_counterfactual(model, x, desired_y, lambda_reg=1.0, num_steps=1000):
    """
    Creates a counterfactual example for a given instance and model.
    Starting from a copy of the instance x, uses the Adam optimizer (with default params) to find a counterfactual of class desired_y.
    Optimizes for a maximum of num_steps steps. Stops the optimization if the predicted class has changed to the desired one.

    Parameters:
        model (MLPClassifier, torch.nn.Module):
            The classifier to generate a counterfactual for. Is a torch.nn.Module, so you can call its forward method using `model()`.
            Its output are logits (unnormalized class probabilities), which have to be normalized by applying a softmax operation.
        x (torch.tensor): The input instance to generate a counterfactual for.
        desired_y (int): The desired output class index to optimize for.
        lambda_reg (float): The regularization strenghts used by the loss function.
        num_steps (int): The maximum number of steps to optimize for.

    Returns:
        counterfactual (torch.tensor): The counterfactual example, of the same shape as the original input x.
        step (int): The last step of the optimization.

    Hint:
        - you can create a copy of a tensor using `.clone()`.
        - you have to convert the desired_y to a one hot representation to be able to compare it with the model output of shape (1, 3).
    """
    
    return None


if __name__ == "__main__":
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

    x, y = ds_test[0]
    x = x.unsqueeze(0)

    print('Running `create_counterfactual` ...')
    cf, step = create_counterfactual(model, x, 0)

    print('Original Instance:')
    print(pd.DataFrame(x.tolist(), columns=dataset.get_input_labels()))
    print(f'\nPrediction: {model.predict(x).item()}')

    print(f'\nCounterfactual for desired class {0}, found after {step} steps:')
    print(pd.DataFrame(cf.tolist(), columns=dataset.get_input_labels()))
    print(f'\nPrediction: {model.predict(cf).item()}')
