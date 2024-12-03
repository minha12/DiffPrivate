from criteria.lpips.lpips import LPIPS


import numpy as np
import torch
from advertorch.utils import clamp_by_pnorm


def perturb_cw(
    xvar,
    yvar,
    predict,
    nb_iter,
    eps,
    eps_iter,
    loss_fn,
    clip_min=-1,
    clip_max=1,
    initial_const=1e-3,
    binary_search_steps=9,
    ord=np.inf,
    lpips_weight=0.0,
    id_weight=1.0,
    l2_weight=1.0,
    targeted_attack=False,
):
    """
    Implementation of the Carlini & Wagner attack.

    :param xvar: input data.
    :param yvar: input labels.
    :param predict: forward pass function.
    :param nb_iter: number of iterations.
    :param eps: maximum distortion.
    :param eps_iter: attack step size.
    :param loss_fn: loss function.
    :param target: target labels for targeted attack. If None, non-targeted attack is performed.
    :param confidence: confidence of adversarial examples.
    :param clip_min: minimum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param initial_const: the initial trade-off constant c to use.
    :param binary_search_steps: number of binary search steps to find the correct c.
    :param minimize: whether to minimize or maximize the loss.
    :param ord: the order of the norm (supporting inf or 2 for now).
    :return: tensor containing the perturbed input.
    """

    # print value range of xvar
    print(xvar.min(), xvar.max())
    # Setup variables
    delta = torch.zeros_like(xvar, requires_grad=True)
    best_attack = xvar.clone()

    # Setup optimizer
    optimizer = torch.optim.Adam([delta], lr=eps_iter)
    lpips_loss = LPIPS()
    best_loss = torch.full((xvar.size(0),), float("inf"), device=xvar.device)

    # Binary search loop
    for binary_step in range(binary_search_steps):
        c = initial_const * 2**binary_step

        for iteration in range(nb_iter):
            optimizer.zero_grad()

            # Calculate adversarial examples
            adv_examples = xvar + delta
            adv_examples = torch.clamp(adv_examples, clip_min, clip_max)

            outputs = predict.extract_feats(adv_examples)
            # l2dist = torch.sqrt(
            #     torch.sum(torch.square(delta.reshape(delta.size(0), -1)), dim=1)
            # )
            idLoss = loss_fn(outputs, yvar, distance_metric=1)
            lpipsLoss = lpips_loss(xvar + delta, xvar)
            if targeted_attack:
                loss = id_weight * idLoss + lpips_weight * lpipsLoss
            else:
                loss = -id_weight * idLoss + lpips_weight * lpipsLoss

            # Backpropagation
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                id_dist_to_orig = predict(xvar + delta, xvar)
            print(id_dist_to_orig)
            # Project delta to be within the allowed epsilon ball
            if ord == np.inf:
                delta.data = torch.clamp(delta, -eps, eps)
            elif ord == 2:
                delta.data = clamp_by_pnorm(delta, ord, eps)

            # Optionally project the perturbation to be exactly epsilon away
            if ord == 2 and eps is not None:
                delta.data = clamp_by_pnorm(delta.data, ord, eps)

            # Keep track of the best attack found
            with torch.no_grad():
                is_better = loss < best_loss
                best_loss = torch.where(is_better, loss, best_loss)
                best_attack = torch.where(
                    is_better.view(-1, 1, 1, 1), adv_examples, best_attack
                )

        # Adjust c based on the results

    return best_attack