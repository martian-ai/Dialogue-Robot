import torch
import math
import torch.nn as nn


def aeq(*args):
    """Assert all arguments have the same value."""
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def sequence_mask(lengths, max_len=None):
    """Create a boolean mask from sequence lengths."""
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def rnn_factory(rnn_type, **kwargs):
    """Rnn factory, Use pytorch version when available."""
    rnn = getattr(nn, rnn_type)(**kwargs)
    return rnn


def gelu(x):
    """Gelu Function.

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def gumbel_softmax(logits, tau=1.0, hard=False, log_mode=True, dim=-1):
    """Gumbel softmax.

    Args:
        logits (_type_): _description_
        tau (float, optional): _description_. Defaults to 1.0.
        hard (bool, optional): _description_. Defaults to False.
        log_mode (bool, optional): _description_. Defaults to True.
        dim (int, optional): _description_. Defaults to -1.

    Returns:
        _type_: _description_
    """
    while (True):
        gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
        if log_mode:
            y_soft = gumbels.log_softmax(dim)
        else:
            y_soft = gumbels.softmax(dim)
        if torch.sum(torch.isnan(y_soft)).item() < 0.01:
            break

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


def gumbel_soft2hard(log_logits, dim=-1):
    """_summary_

    Args:
        log_logits (_type_): _description_
        dim (int, optional): _description_. Defaults to -1.

    Returns:
        _type_: _description_
    """
    y_soft = log_logits.exp()
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(log_logits).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
    return ret