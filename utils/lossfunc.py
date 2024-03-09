import torch.nn as nn
import torch.nn.functional as F

def mseloss(predict_q, target_q):
    '''
    loss function to regress the predicted Q value
    '''
    return F.mse_loss(predict_q, target_q)