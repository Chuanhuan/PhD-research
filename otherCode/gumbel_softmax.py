import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False):
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = torch.zeros_like(y)
        y_hard.scatter_(1, y.argmax(dim=-1, keepdim=True), 1.0)
        y = (y_hard - y).detach() + y
    return y


class GumbelSoftmaxNet(nn.Module):
    def __init__(self, input_dim, output_dim, temperature):
        super(GumbelSoftmaxNet, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.temperature = temperature

    def forward(self, x):
        logits = self.linear(x)
        y = gumbel_softmax(logits, self.temperature, hard=True)
        return y


# Example usage
if __name__ == "__main__":
    input_dim = 10
    output_dim = 5
    temperature = 0.5

    model = GumbelSoftmaxNet(input_dim, output_dim, temperature)
    x = torch.randn(2, input_dim)
    y = model(x)
    print(y)
