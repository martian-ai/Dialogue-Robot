import torch.nn as nn
import torch

num_tags = 3

CUDA = torch.cuda.is_available()
randn = lambda *x: torch.randn(*x).cuda() if CUDA else torch.randn

# print(torch.randn(num_tags, num_tags))
# print(randn(num_tags, num_tags))
# nn.Parameter(randn(num_tags, num_tags))
print(nn.Parameter(torch.randn(num_tags, num_tags)))