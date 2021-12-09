import torch
import torch.nn as nn
import kernel


class GCNConv(nn.Module):
    def __init__(self, A_hat, size_1, size_2, bias=True, device='cuda'):
        super(GCNConv, self).__init__()
        self.A_hat = torch.tensor(A_hat, requires_grad=False).float().to(device)
        # Weight Matrix
        self.weight = nn.parameter.Parameter(torch.FloatTensor(size_1, size_2))
        nn.init.xavier_uniform_(self.weight)

        if bias:
            self.bias = nn.parameter.Parameter(torch.FloatTensor(size_2))
            nn.init.zeros_(self.bias)

    def forward(self, X):
        X = torch.mm(X, self.weight)
        if self.bias is not None:
            X = (X + self.bias)

        res = kernel.product(self.A_hat.cpu().detach().numpy(), X.cpu().detach().numpy())
        # print("Kernel finished.")

        return torch.FloatTensor(res).to('cuda')
        # return torch.mm(self.A_hat, X)


class GCNNet(nn.Module):
    def __init__(self, A_hat, X_dim, hidden_size_1, hidden_size_2, num_classes, bias=True): # X_size = num features
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(A_hat, X_dim, hidden_size_1, bias)
        self.ReLU = nn.ReLU()
        self.conv2 = GCNConv(A_hat, hidden_size_1, hidden_size_2, bias)
        self.linear = nn.Linear(hidden_size_2, num_classes)
        
    def forward(self, X):
        out = self.conv1(X)
        out = self.ReLU(out)
        out = self.conv2(out)
        out = self.linear(out)
        return out