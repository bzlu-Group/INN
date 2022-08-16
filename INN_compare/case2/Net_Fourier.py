import numpy as np 
import torch
import torch.nn as nn

class DeepFourier(nn.Module):
    def __init__(self,input_dim=2,output_dim=1,m=40,device='cpu',actv=nn.Tanh()):
        super(DeepFourier, self).__init__()
        self.actv=actv
        self.linear1 =nn.Linear(m,m)
        self.linear2 =nn.Linear(m,m)
        self.linear3 = nn.Linear(m,m)
        self.linear4 =nn.Linear(m,m)
        self.linear_output = nn.Linear(m,output_dim)

        self.Fourier=self.__ff_trans(m,var=5).to(device)

    def __ff_trans(self,m,var):
        mean = (0, 0)
        cov = np.array([[var**2, 0], [0, var**2]])
        x = np.random.multivariate_normal(mean, cov, (m//2,), 'raise').T   # nx2
        x = torch.tensor(x).float()

        return x       

    def forward(self, x):
        y= torch.hstack((torch.sin(x@self.Fourier),torch.cos(x@self.Fourier)))
        y = y + self.actv(self.linear2(self.actv(self.linear1(y))))
        y = y + self.actv(self.linear4(self.actv(self.linear3(y))))

        output = self.linear_output(y)
        return output

