import torch


def mae(pred, target):
    return torch.mean(torch.abs(pred - target)).item()


def mse(pred, target):
    return torch.mean((pred - target) ** 2).item()


def rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def psnr(pred, target, max_pixel=1.0):
    mse_value = torch.mean((pred - target) ** 2)
    if mse_value == 0:
        return 100.0
    return (20 * torch.log10(torch.tensor(max_pixel)) - 10 * torch.log10(mse_value)).item()