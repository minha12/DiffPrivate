from criteria.lpips.lpips import LPIPS


import numpy as np
import torch
from advertorch.utils import batch_clamp, batch_l1_proj, batch_multiply, clamp, clamp_by_pnorm, normalize_by_pnorm


def perturb_pgd(
    xvar,
    yvar,
    predict,
    nb_iter,
    eps,
    eps_iter,
    loss_fn,
    attack_iter=1,
    repeat_times=1,
    delta_init=None,
    minimize=False,
    ord=np.inf,
    clip_min=0.0,
    clip_max=1.0,
    alpha=1.0,
    l1_sparsity=None,
    lpips_weight=0.0,
    id_weight=1.0,
    targeted_attack=False,
):
    """
    Iteratively maximize the loss over the input. It is a shared method for
    iterative attacks including IterativeGradientSign, LinfPGD, etc.

    :param xvar: input data.
    :param yvar: input labels.
    :param predict: forward pass function.
    :param nb_iter: number of iterations.
    :param eps: maximum distortion.
    :param eps_iter: attack step size.
    :param loss_fn: loss function.
    :param delta_init: (optional) tensor contains the random initialization.
    :param minimize: (optional bool) whether to minimize or maximize the loss.
    :param ord: (optional) the order of maximum distortion (inf or 2).
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param l1_sparsity: sparsity value for L1 projection.
                  - if None, then perform regular L1 projection.
                  - if float value, then perform sparse L1 descent from
                    Algorithm 1 in https://arxiv.org/pdf/1904.13000v1.pdf
    :return: tensor containing the perturbed input.
    """
    if delta_init is not None:
        delta = delta_init
    else:
        delta = torch.zeros_like(xvar)

    if yvar is None:  # non-targeted attack
        with torch.no_grad():
            yvar = predict.extract_feats(xvar)
    delta.requires_grad_()
    lpips_loss = LPIPS()
    for j in range(nb_iter):

        for k in range(repeat_times):
            outputs = predict.extract_feats(xvar + delta)
            for ii in range(attack_iter):
                idLoss = loss_fn(outputs, yvar)
                lpipsLoss = lpips_loss(xvar + delta, xvar)
                if targeted_attack:
                    loss = -id_weight * idLoss - lpips_weight * lpipsLoss
                else:
                    loss = id_weight * idLoss - lpips_weight * lpipsLoss

                if minimize:
                    loss = -loss

                loss.backward()
                with torch.no_grad():
                    iddist_to_orig = predict(xvar + delta, xvar)
                print(iddist_to_orig.item())

                delta_old = delta.data

                if ord == np.inf:
                    grad_sign = delta.grad.data.sign()
                    delta.data = delta.data + batch_multiply(eps_iter, grad_sign)
                    delta.data = batch_clamp(eps, delta.data)
                    delta.data = (
                        clamp(xvar.data + delta.data, clip_min, clip_max) - xvar.data
                    )

                elif ord == 2:
                    grad = delta.grad.data
                    grad = normalize_by_pnorm(grad)
                    delta.data = delta.data + batch_multiply(eps_iter, grad)
                    delta.data = (
                        clamp(xvar.data + delta.data, clip_min, clip_max) - xvar.data
                    )
                    if eps is not None:
                        delta.data = clamp_by_pnorm(delta.data, ord, eps)

                elif ord == 1:
                    grad = delta.grad.data
                    abs_grad = torch.abs(grad)

                    batch_size = grad.size(0)
                    view = abs_grad.reshape(batch_size, -1)
                    view_size = view.size(1)
                    if l1_sparsity is None:
                        vals, idx = view.topk(1)
                    else:
                        vals, idx = view.topk(
                            int(np.round((1 - l1_sparsity) * view_size))
                        )

                    out = torch.zeros_like(view).scatter_(1, idx, vals)
                    out = out.view_as(grad)
                    grad = grad.sign() * (out > 0).float()
                    grad = normalize_by_pnorm(grad, p=1)
                    delta.data = delta.data + batch_multiply(eps_iter, grad)

                    delta.data = batch_l1_proj(delta.data.cpu(), eps)
                    delta.data = delta.data.to(xvar.device)
                    delta.data = (
                        clamp(xvar.data + delta.data, clip_min, clip_max) - xvar.data
                    )
                else:
                    error = "Only ord = inf, ord = 1 and ord = 2 have been implemented"
                    raise NotImplementedError(error)

                delta.data = alpha * delta.data + (1 - alpha) * delta_old

                delta.grad.data.zero_()

    x_adv = clamp(xvar + delta, clip_min, clip_max)

    return x_adv