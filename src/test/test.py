import torch

from ..dropgrad import DropGrad


def test_dropgrad():
    
    data = torch.randn(32, 10)
    
    mlp = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 30),
        torch.nn.ReLU(),
        torch.nn.Linear(30, 10)
    )
    loss = torch.nn.MSELoss()
    
    opt = DropGrad(torch.optim.Adam(mlp.parameters(), lr=0.001), drop_rate=0.1)
    
    pred = mlp(data)
    loss_pred = loss(pred, data)
    
    loss_pred.backward()
    
    total_params = 0
    total_zero_params = 0
    for p in mlp.parameters():
        if p.grad is not None:
            total_params += p.grad.data.numel()
            total_zero_params += torch.sum(torch.abs(p.grad.data) < 1e-10).item()
            
    print(f"Zero grad params before step: {total_zero_params / total_params}")
    
    opt.step()
    
    total_params = 0
    total_zero_params = 0
    for p in mlp.parameters():
        if p.grad is not None:
            total_params += p.grad.data.numel()
            total_zero_params += torch.sum(torch.abs(p.grad.data) < 1e-10).item()
            
    print(f"Zero grad params before step: {total_zero_params / total_params}")
