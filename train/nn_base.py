
import torch 
import torch.nn as nn
import numpy as np

precision = np.float32

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()

def to_tensor(a):
    a = np.array(a, precision)
    return torch.from_numpy(a)
    
def to_tensors_tuple(nas):
    ts = []
    for a in nas:
        ts.append(nn_base.to_tensor(a))
    return tuple(ts)

class MLP(nn.Module):
    def __init__(self,
                 input_dim:[tuple,list,int],
                 hidden_dim:[tuple,list,int],
                 output_dim:[tuple,list,int]):
        super().__init__()
        if isinstance(input_dim, int): input_dim = [input_dim]
        if isinstance(hidden_dim, int): hidden_dim = [hidden_dim]
        if isinstance(output_dim, int): output_dim = [output_dim]
        self.input_dim = np.prod(input_dim)
        self.output_dim = np.prod(output_dim)
        hidden_dim = np.array(hidden_dim)
        self.hidden_dim = [np.prod(x) for x in hidden_dim]
        self.trunk = self.mlp(self.input_dim, self.hidden_dim, self.output_dim)
        self.apply(weight_init)

    def mlp(self, input_dim, hidden_dim, output_dim):
        hidden_depth = len(hidden_dim)
        if hidden_depth == 0:
            mods = [nn.Linear(input_dim, output_dim)]
        else:
            mods = [nn.Linear(input_dim, hidden_dim[0]), nn.ReLU()]
            for i in range(hidden_depth - 1):
                mods += [nn.Linear(hidden_dim[i], hidden_dim[i+1]), nn.ReLU()]
            mods.append(nn.Linear(hidden_dim[-1], output_dim))
        trunk = nn.Sequential(*mods)
        return trunk

    def forward(self, x):
        x = x.view(-1, np.prod(self.input_dim))
        return self.trunk(x)


if __name__ == "__main__":
    losses = []
    for t in range(1000):
        m = MLP(2,[32],1)
        optimizer = torch.optim.Adam(m.parameters(), lr=1e-1, betas=(0.9,0.999), eps=1e-8, weight_decay=0, amsgrad=False)
        xor_in = to_tensor([[0,0],[0,1],[1,0],[1,1]])
        xor_out = to_tensor([[0],[1],[1],[0]])
        i = 0
        while i == 0 or len(losses) == 0 and i < 20 or len(losses) > 0 and loss.item() > np.mean(losses) and i < 100:
            i += 1
            predict = m.forward(xor_in)
            loss = nn.functional.mse_loss(predict, xor_out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if len(losses) == 0 or loss.item() < np.min(losses) or loss.item() > np.max(losses):
            print(to_np(predict).tolist())
            print("epoch", t, "step", i, "loss", loss.item())
        losses += [loss.item()]
    print(np.min(losses), np.mean(losses), np.max(losses))

