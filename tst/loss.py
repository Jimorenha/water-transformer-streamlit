import torch
import torch.nn as nn
from numpy import array
import numpy as np


class OZELoss(nn.Module):
    """Custom loss for TRNSys metamodel.

    Compute, for temperature and consumptions, the intergral of the squared differences
    over time. Sum the log with a coeficient ``alpha``.

    .. math::
        \Delta_T = \sqrt{\int (y_{est}^T - y^T)^2}

        \Delta_Q = \sqrt{\int (y_{est}^Q - y^Q)^2}

        loss = log(1 + \Delta_T) + \\alpha \cdot log(1 + \Delta_Q)

    Parameters:
    -----------
    alpha:
        Coefficient for consumption. Default is ``0.3``.
    """

    def __init__(self, reduction: str = 'mean', alpha: float = 0.3):
        super().__init__()

        self.alpha = alpha
        self.reduction = reduction

        self.base_loss = nn.MSELoss(reduction=self.reduction)

    def forward(self,
                y_true: torch.Tensor,
                y_pred: torch.Tensor) -> torch.Tensor:
        """Compute the loss between a target value and a prediction.

        Parameters
        ----------
        y_true:
            Target value.
        y_pred:
            Estimated value.

        Returns
        -------
        Loss as a tensor with gradient attached.
        """
        y_pred_1 = y_pred[..., :-1]
        y_true_1 = y_true[..., :-1]
        y_pred1 = y_pred[..., -1]
        y_true1 = y_true[..., -1]
        delta_Q = self.base_loss(y_pred[..., :-9], y_true[..., :-9])
        delta_T = self.base_loss(y_pred[..., -9:], y_true[..., -9:])
        # delta_Q = torch.from_numpy(array([self.base_loss(y_true[...,i :-1], y_pred[...,i ,:-1]) for i in range(y_true.shape[1])],dtype=np.float32)).mean()
        # delta_T = torch.from_numpy(array([self.base_loss(y_true[..., i,-1], y_pred[...,i, -1]) for i in range(y_true.shape[1])],dtype=np.float32)).mean()
        if self.reduction == 'none':
            delta_Q = delta_Q.mean(dim=(1, 2))
            delta_T = delta_T.mean(dim=(1))

        return torch.log(1 + delta_T) + self.alpha * torch.log(1 + delta_Q)

class AccurcyLoss(nn.Module):
    """Custom loss for TRNSys metamodel.

    Compute, for temperature and consumptions, the intergral of the squared differences
    over time. Sum the log with a coeficient ``alpha``.

    .. math::
        Delta_T = MSE(y_pre,y_true)
        Delta_Q = MSE(y_true,0)
        loss = Delta_T/Delta_Q

    Parameters:
    -----------
    alpha:
        Coefficient for consumption. Default is ``0.3``.
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()

        self.reduction = reduction

        self.base_loss = nn.MSELoss(reduction=self.reduction)

    def forward(self,
                y_true: torch.Tensor,
                y_pred: torch.Tensor) -> torch.Tensor:
        """Compute the loss between a target value and a prediction.

        Parameters
        ----------
        y_true:
            Target value.
        y_pred:
            Estimated value.

        Returns
        -------
        Loss as a tensor with gradient attached.
        """
        y_shape = y_pred.shape
        zero_y = torch.zeros(size=y_shape,device=y_pred.device)
        delta_Q = self.base_loss(y_pred[..., :], y_true[..., :])
        delta_T = self.base_loss(y_true[..., :], zero_y)
        # delta_Q = torch.from_numpy(array([self.base_loss(y_true[...,i :-1], y_pred[...,i ,:-1]) for i in range(y_true.shape[1])],dtype=np.float32)).mean()
        # delta_T = torch.from_numpy(array([self.base_loss(y_true[..., i,-1], y_pred[...,i, -1]) for i in range(y_true.shape[1])],dtype=np.float32)).mean()
        if self.reduction == 'none':
            delta_Q = delta_Q.mean(dim=(1, 2))
            delta_T = delta_T.mean(dim=(1))
        accurcy = delta_Q/delta_T

        return accurcy