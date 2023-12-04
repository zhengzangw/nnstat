from .core import StateDict


def compute_update(w0: StateDict, w1: StateDict, alpha: float = 1.0) -> StateDict:
    """Compute the update of the optimizer by :math:`(w_1 - w_0)/\\alpha`.

    Args:
        w0 (StateDict): initial state dict
        w1 (StateDict): final state dict
        alpha (float, optional): step size. Defaults to 1.0.

    Returns:
        StateDict: update
    """
    assert w0.is_compatible(w1)
    return (w1 - w0) / alpha


def compute_adam_update(m: StateDict, v: StateDict, eps: float = 1e-6):
    """Compute the update of Adam by :math:`m / (\\sqrt{v} + \\epsilon)`.

    Args:
        m (StateDict): first moment
        v (StateDict): second moment
        eps (float, optional): epsilon. Defaults to 1e-6.

    Returns:
        StateDict: update
    """
    return m / (v.sqrt() + eps)


def compute_trust_ratio():
    """From `Large Batch Training of Convolutional Networks <https://arxiv.org/abs/1708.03888>`_
    """
    pass

# def compute_trust_ratio(
#     w0: Union[NetDict, nn.Module],
#     w1: Union[NetDict, nn.Module],
#     lr: float = 1.0,
#     display=True,
#     pattern: Union[str, List[str]] = None,
# ):
#     # |w| / (|w - w'| / lr)
#     if isinstance(w0, nn.Module):
#         w0 = get_weight(w0)
#     if isinstance(w1, nn.Module):
#         w1 = get_weight(w1)
#     assert w0.is_compatible(w1)

#     columns = ["layer_name", "trust_ratio"]
#     ret = pd.DataFrame(columns=columns)

#     layer_name = []
#     trust_ratio = []
#     for i, key in enumerate(w0.keys()):
#         if pattern is not None and not any([inc in key for inc in pattern]):
#             continue
#         layer_name.append(key)
#         tr = (lr * w0[key].norm(2) / (w0[key] - w1[key]).norm(2)).item()
#         trust_ratio.append(tr)
#     ret["layer_name"] = layer_name
#     ret["trust_ratio"] = trust_ratio

#     if display:
#         print_title("Trust ratio")
#         print(ret)
#     else:
#         return ret


# def compute_noise_scale(
#     model: nn.Module,
#     forward_backward_closure: Callable,
#     micro_batch_size: int,
#     num_iters: int = 100,
#     num_average: int = 1,
#     display=True,
#     save_plot=False,
# ):
#     """
#     Args:
#         forward_backward_closure: closure should do: get data, forward, backward
#             Example:
#                 X, Y = get_batch()
#                 logits, loss = model(X, Y)
#                 loss.backward()
#             Do not zero_grad() in the closure
#     """
#     L1_norms = np.zeros(num_iters)
#     L2_2_norms = np.zeros(num_iters)
#     bs = np.arange(1, num_iters + 1) * micro_batch_size

#     for k in range(num_average):
#         cur_bs = 0
#         model.zero_grad()
#         for i in tqdm(range(num_iters)):
#             cur_bs += micro_batch_size
#             forward_backward_closure()
#             L1_norm = get_grad(model).norm(1) / cur_bs
#             L2_norm_2 = (get_grad(model).norm(2) / cur_bs) ** 2
#             L1_norms[i] += L1_norm
#             L2_2_norms[i] += L2_norm_2
#     model.zero_grad()
#     L1_norms /= num_average
#     L2_2_norms /= num_average

#     columns = ["method", "b_noise", "g_2", "trace_sigma"]
#     ret = []

#     # by curve fitting
#     def func(bs_inv, g_2, trace_sigma):
#         return g_2 + bs_inv * trace_sigma

#     popt, pcov = curve_fit(func, 1 / bs, L2_2_norms)
#     g_2, trace_sigma = popt
#     b_noise = trace_sigma / g_2
#     ret.append(
#         dict(method="curve_fit", b_noise=b_noise, g_2=g_2, trace_sigma=trace_sigma)
#     )

#     # by two point fitting
#     b_small_index = (num_iters - 1) // 2
#     b_big_index = num_iters - 1
#     b_small, g_2_small = bs[b_small_index], L2_2_norms[b_small_index]
#     b_big, g_2_big = bs[b_big_index], L2_2_norms[b_big_index]

#     g_2 = (g_2_big * b_big - g_2_small * b_small) / (b_big - b_small)
#     trace_sigma = (g_2_small - g_2_big) / (1 / b_small - 1 / b_big)
#     b_noise = trace_sigma / g_2
#     ret.append(
#         dict(method="two_point", b_noise=b_noise, g_2=g_2, trace_sigma=trace_sigma)
#     )

#     ret = pd.DataFrame(ret, columns=columns)

#     if save_plot:
#         plot_line(
#             pd.DataFrame(dict(batch_size=bs, L2_2_norms=L2_2_norms)),
#             name="L2_2_vs_bs",
#         )
#         plot_line(
#             pd.DataFrame(dict(inv_batch_size=1 / bs, L2_2_norms=L2_2_norms)),
#             name="L2_2_vs_inv_bs",
#         )

#     # save and print
#     if display:
#         print_title("Noise scale")
#         print(ret)
#     else:
#         return ret
