import torch as th
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from depthwise import DepthwiseNet
from torch.nn.utils import weight_norm
import numpy as np

class ADDSTCN(nn.Module):
    def __init__(self, target, input_size, num_levels, kernel_size, cuda, dilation_c,
                 lasso_lambda=1e-6):
        super(ADDSTCN, self).__init__()
        
        self.device = th.device("cuda" if cuda else "cpu")

        self.lasso_lambda = lasso_lambda
        
        self.target=target
        self.dwn = DepthwiseNet(self.target, input_size, num_levels, kernel_size=kernel_size, dilation_c=dilation_c)
        self.pointwise = nn.Conv1d(input_size, 1, 1, device=self.device)

        #self._attention = th.ones(input_size,1)
        self.fs_attention_logits = nn.Parameter(th.ones(input_size, 1, device=self.device))
        
        if cuda:
            self.dwn = self.dwn.cuda()
                  
    def init_weights(self) -> None:
        self.pointwise.weight.data.normal_(0, 0.1)       
        
    def forward(self, x):
        
        # new variable for LASSO attention scores interpretability
        attention_scores = th.sigmoid(self.fs_attention_logits)
        
        #y1=self.dwn(x*F.softmax(self.fs_attention, dim=0))
        y1=self.dwn(x*attention_scores)
        y1 = self.pointwise(y1) 
        return y1.transpose(1,2)
    
    def attention_regularization(self, p=1):
        """L1 penalty on attention scores [0,1]"""
        attention_scores = th.sigmoid(self.fs_attention_logits)
        return self.lasso_lambda * th.norm(attention_scores, p=1)
    
    def get_sparsity_stats(self):
        scores = th.sigmoid(self.fs_attention_logits).detach().squeeze()
        
        return {
            'mean_attention': scores.mean().item(),
            'near_zero': (scores < 0.1).sum().item(),  # "Spente"
            'active': (scores > 0.5).sum().item(),     # "Attive"
            'active_idx': (scores > 0.5).nonzero(as_tuple=True)[0].cpu().numpy(),
            'sparsity_ratio': (scores < 0.1).float().mean().item()
        }
        
    def get_attention_scores(self):
        """Returns attention scores as a numpy array."""
        scores = th.sigmoid(self.fs_attention_logits).detach().cpu().numpy()
        return scores.squeeze()