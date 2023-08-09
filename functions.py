from random import Random
from typing import List, Tuple
from math import sqrt
from matplotlib import pyplot as plt


SEED = 5
random_gen = Random()

## Generate Points
def gen_pts_(n: int) -> Tuple[List[float], List[float]]:
    """generates n random points in the unit square, using for loops
    Args:
        n (int): number of points to generate
    Returns:
        Tuple[List[float], List[float]]: list of x coordinates and list of y coordinates
    """
    random_gen.seed(SEED)
    lst_x, lst_y = [], []
    for _ in range(n):
        lst_x.append(random_gen.uniform(a=0, b=1))
    for _ in range(n):
        lst_y.append(random_gen.uniform(a=0, b=1))
    # print(random_gen.random())
    return lst_x, lst_y

def gen_pts_list_comp_(n: int) -> List[List[float]]:
    """generates n random points in the unit square, using list comprehension
    Args:
        n (int): number of points to generate
    Returns:
        Tuple[List[float], List[float]]: list of x coordinates and list of y coordinates
    """

    random_gen.seed(SEED)
    return [[random_gen.uniform(a=0, b=1) for _ in range(n)],[random_gen.uniform(a=0, b=1) for _ in range(n)]] 


## Calculate Loss

def loss(x_p: float, y_p: float, data_x: List[float], data_y: List[float]) -> float:
    """calculates the loss for a given point (x_p, y_p) and a set of points (data_x, data_y)
    args:
        x_p (float): x coordinate of the point
        y_p (float): y coordinate of the point
        data_x (List[float]): list of x coordinates of the points
        data_y (List[float]): list of y coordinates of the points
    returns:
        float: loss, i.e. the average distance between the point and the points in the dataset
    """
    loss = (1 / len(data_x)) * sum(
        [sqrt((x_i - x_p) ** 2 + (y_i - y_p) ** 2) for x_i, y_i in zip(data_x, data_y)]
    )
    return loss


def calc_grad(x_p: float, y_p: float, data_x: List[float], data_y: List[float]) -> Tuple[float, float]:
    """calculates the gradient of the loss function for a given point (x_p, y_p) and a set of points (data_x, data_y)
        using closed form solution
    Args:
        x_p (float): x coordinate of the point
        y_p (float): y coordinate of the point
        data_x (List[float]): list of x coordinates of the points
        data_y (List[float]): list of y coordinates of the points

    Returns:
        Tuple[float, float]: gradient of the loss function with respect to x and y
    """
    sum_x, sum_y = 0, 0
    for x_i, y_i in zip(data_x, data_y):
        inv_sqrt = ((x_i - x_p) ** 2 + (y_i - y_p) ** 2) ** -0.5
        sum_x += inv_sqrt * (x_i - x_p)
        sum_y += inv_sqrt * (y_i - y_p)
    return sum_x / -len(data_x), sum_y / -len(data_y)


def Gradient_hist(
    x_p: float,
    y_p: float,
    data_x: List[float],
    data_y: List[float],
    EPOCHS: int = 1000,
    DELTA: float = 0.01,
    H: float = 0.001,
    CLOSE_FROM: bool = False,
):

    """performs gradient descent on the loss function numerically, using for loops
    Args:
        x_p (float): x coordinate of the point
        y_p (float): y coordinate of the point
        data_x (List[float]): list of x coordinates of the points
        data_y (List[float]): list of y coordinates of the points
        EPOCHS (int, optional): number of epochs. Defaults to 1000.
        DELTA (float, optional): learning rate. Defaults to 0.01.
        H (float, optional): step size for numerical differentiation. Defaults to 0.001.

    Returns:
        Tuple[List[float], List[float], List[float], List[float], List[float]]: list of losses, list of dl_dx values, list of dl_dy values, list of x_p values, list of y_p values
    """
    epoch_losses = []
    dl_dx_values = []
    dl_dy_valyes = []
    xp_values = [x_p]
    yp_values = [y_p]

    for _ in range(EPOCHS):

        epoch_losses.append(loss(x_p, y_p, data_x, data_y))
        
        if CLOSE_FROM:
            dloss_dx, dloss_dy = calc_grad(x_p, y_p, data_x, data_y)
        else:
            dloss_dx = (loss(x_p + H, y_p, data_x, data_y) - loss(x_p, y_p, data_x, data_y)) / H
            dloss_dy = (loss(x_p, y_p + H, data_x, data_y) - loss(x_p, y_p, data_x, data_y)) / H

        dl_dx_values.append(dloss_dx)
        dl_dy_valyes.append(dloss_dy)

        x_p -= DELTA * dloss_dx
        y_p -= DELTA * dloss_dy
        xp_values.append(x_p)
        yp_values.append(y_p)

    return epoch_losses, dl_dx_values, dl_dy_valyes, xp_values, yp_values


def comp_plot(close_form, limits, y_low= 0, y_high=5, title ="x_p values"):
    """compares the results of the closed form solution and the numerical solution"""
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].set_title(f"Updating {title} using closed form Solution")
    ax[0].plot(close_form)
    ax[0].set_ylim(y_low, y_high)
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel(title)

    ax[1].set_title(f"updating {title} using limits")
    ax[1].plot(limits)
    ax[1].set_ylim(y_low, y_high)
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("title")