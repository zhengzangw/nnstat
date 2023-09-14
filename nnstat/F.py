from pprint import pprint
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .core import *


def compute_stats(
    model: nn.Module,
    optimizer: torch.optim.Optimizer = None,
    optimizer_keys=(),
    display=True,
):
    weight = NetDict.get_weight(model)
    grad = NetDict.get_grad(model)

    # optimizer state
    optimizer_state = []
    for optimizer_key in optimizer_keys:
        optimizer_state.append(
            NetDict.get_optimizer_state(model, optimizer, optimizer_key)
        )

    if display:
        print_title("Weight")
        weight.describe()
        print_title("Grad")
        grad.describe()
        for i, optimizer_key in enumerate(optimizer_keys):
            print_title(f"Optimizer state {optimizer_key}")
            optimizer_state[i].describe()

    return dict(weight=weight, grad=grad, optimizer_state=optimizer_state)


def compute_trust_ratio(
    w0: Union[NetDict, nn.Module],
    w1: Union[NetDict, nn.Module],
    include: Union[str, List[str]] = None,
    display=True,
):
    # |w| / |w - w'|
    if isinstance(w0, nn.Module):
        w0 = get_weight(w0)
    if isinstance(w1, nn.Module):
        w1 = get_weight(w1)
    assert w0.is_compatible(w1)

    trust_ratios = dict()
    for key in w0.keys():
        if include is not None and not any([inc in key for inc in include]):
            continue
        trust_ratios[key] = (w0[key].norm(2) / (w0[key] - w1[key]).norm(2)).item()
    if display:
        print_title("Trust ratio")
        tab = PrettyTable()
        tab.field_names = ["layer name", "trust ratio"]
        tab.align = "l"
        for key, value in trust_ratios.items():
            tab.add_row([key, f"{value:.5g}"])
        print(tab)
    return trust_ratios


def compute_noise_scale(
    model: nn.Module,
    forward_backward_closure: Callable,
    micro_batch_size: int,
    num_iters: int = 100,
    num_average: int = 1,
    display=True,
    save_plot=False,
):
    """
    Args:
        forward_backward_closure: closure should do: get data, forward, backward
            Example:
                X, Y = get_batch()
                logits, loss = model(X, Y)
                loss.backward()
            Do not zero_grad() in the closure
    """
    L1_norms = np.zeros(num_iters)
    L2_2_norms = np.zeros(num_iters)
    bs = np.arange(1, num_iters + 1) * micro_batch_size

    for k in range(num_average):
        cur_bs = 0
        model.zero_grad()
        for i in tqdm(range(num_iters)):
            cur_bs += micro_batch_size
            forward_backward_closure()
            L1_norm = get_grad(model).norm(1) / cur_bs
            L2_norm_2 = (get_grad(model).norm(2) / cur_bs) ** 2
            L1_norms[i] += L1_norm
            L2_2_norms[i] += L2_norm_2
    compute_stats(model)
    model.zero_grad()
    L1_norms /= num_average
    L2_2_norms /= num_average

    # by curve fitting
    from scipy.optimize import curve_fit

    def func(bs_inv, g_2, trace_sigma):
        return g_2 + bs_inv * trace_sigma

    popt, pcov = curve_fit(func, 1 / bs, L2_2_norms)
    g_2, trace_sigma = popt
    b_noise = trace_sigma / g_2
    results_curve_fit = dict(g_2=g_2, trace_sigma=trace_sigma, b_noise=b_noise)

    # by two point fitting
    b_small_index = (num_iters - 1) // 2
    b_big_index = num_iters - 1
    b_small, g_2_small = bs[b_small_index], L2_2_norms[b_small_index]
    b_big, g_2_big = bs[b_big_index], L2_2_norms[b_big_index]

    g_2 = (g_2_big * b_big - g_2_small * b_small) / (b_big - b_small)
    trace_sigma = (g_2_small - g_2_big) / (1 / b_small - 1 / b_big)
    b_noise = trace_sigma / g_2
    results_two_point = dict(g_2=g_2, trace_sigma=trace_sigma, b_noise=b_noise)

    # save and print
    ret = dict(
        curve_fit=results_curve_fit,
        two_point=results_two_point,
    )
    if display:
        print_title("Noise scale")
        pprint(ret)

    if save_plot:
        plot_curve(
            bs,
            dict(data=L2_2_norms),
            name="L2_2_vs_bs",
            x_label="batch size",
            y_label="grad L2 norm squared",
        )
        plot_curve(
            1 / bs,
            dict(data=L2_2_norms),
            name="L2_2_vs_inv_bs",
            x_label="1 / batch size",
            y_label="grad L2 norm squared",
        )

    return ret
