import torch


def IoU(mask1, mask2):
    mask1 = mask1.bool()
    mask2 = mask2.bool()
    intersection = torch.logical_and(mask1, mask2).sum().float()
    union = torch.logical_or(mask1, mask2).sum().float()
    if union.item() == 0:
        return torch.tensor(0.0, device=mask1.device)
    return intersection / union

