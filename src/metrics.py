import torch
import numpy as np

def MSE(y_true, y_pred, occupation=None, idx_label=None, reduction='mean'):
    # Select output labels
    idx_label = idx_label or torch.arange(y_true.shape[-1])

    # Compute squared difference
    diff = torch.pow(y_true[..., idx_label]-y_pred[..., idx_label], 2)

    if occupation is not None:
        # Add dimension for broacasting
        occupation = occupation.unsqueeze(-1)

        # Mask with occupation
        diff = diff * occupation

    if reduction == 'mean':
        return torch.mean(diff).item()
    elif reduction == 'none':
        return torch.mean(diff, dim=(1, 2)).numpy()

def CORR(y_true,y_pred):
    mt=torch.mean(y_true,0)
    mp=torch.mean(y_pred,0)
    tm=torch.sub(y_true,mt)
    pm=torch.sub(y_pred,mp)
    r_num=torch.sum(tm*pm)
    t_square_sum=torch.sum(tm*tm)
    p_square_sum=torch.sum(pm*pm)
    r_den=torch.sqrt(t_square_sum*p_square_sum)
    r=r_num/r_den
    return r

def RAE(y_true,y_pred):
    return torch.sum(torch.abs(y_true - y_pred)) / (torch.sum(torch.abs(y_true - torch.mean(y_true,0))))
def RSE(y_true,y_pred):
    return torch.sum(torch.square(y_true - y_pred)) / (torch.sum(torch.square(y_true - torch.mean(y_true,0))))
def RRSE(y_true,y_pred):
    return torch.sqrt(torch.sum(torch.square(y_true - y_pred)) / torch.sum(torch.square(y_true - torch.mean(y_true,0))))