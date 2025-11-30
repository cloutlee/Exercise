import torch.nn as nn
import torch


# print(torch.zeros(5))
# print(torch.ones(5))
# print(torch.full((5,), 3.14))
# print(torch.arange(0, 10, 2))
# print(torch.linspace(0, 1, steps=5))
# print(torch.eye(3))
# print(torch.rand(3, 3))
# print(torch.randn(3, 3))
# print(torch.randint(0, 10, (3, 3)))
# print(torch.empty(3, 3))
# print(torch.tensor([1.0, 2.0, 3.0]))



# a = torch.randn(5)
# print(a)
# re = nn.ReLU()
# output = re(a)
# print(output)



bce_loss = nn.BCELoss()
predictions = torch.tensor([0.8, 0.2, 0.9, 0.1])
targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
loss = bce_loss(predictions, targets)
print(f"BCELoss: {loss.item()}")


