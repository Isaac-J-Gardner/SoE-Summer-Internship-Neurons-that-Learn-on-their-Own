import torch

a = torch.rand(5, 1, 1, 2)
print(a)
a = a.view(5, -1)
m = torch.mean(a, dim=1)
print(m)