import torch
import torch.nn as nn
import torch.utils.data as data


class MyOne(torch.nn.Module):
    def __init__(self, D_in, H1, H2, H3, H4, D_out):
        super().__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear3 = torch.nn.Linear(H2, H3)
        self.linear4 = torch.nn.Linear(H3, H4)
        self.linear5 = torch.nn.Linear(H4, D_out)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))

        x = torch.nn.functional.relu(self.linear3(x))

        x = torch.nn.functional.relu(self.linear4(x))

        x = self.linear5(x)

        return x
