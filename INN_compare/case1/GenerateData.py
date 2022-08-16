# -*- coding: utf-8 -*-
import torch
import numpy as np

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


pi=float(np.pi)
class Data(object):
    def __init__(self,r0,L,box,device):
        self.r0=r0
        self.L=torch.tensor(L).to(device)
        self.box=torch.tensor(box).to(device)
        self.device=device

    def SampleFromGamma(self,num):
        L=self.L
        psi=2*torch.rand(num,device=self.device).view(-1,1)   #[0,2pi]
        x=L[0]+self.r0*torch.cos(psi*pi)
        y=L[1]+self.r0*torch.sin(psi*pi)   
        X=torch.cat((x,y),dim=1) 
           
        f_direction=(X-self.L)/self.r0
        
        return X.to(self.device),f_direction.to(self.device)
    


    def SampleFromDomain(self,num):
        L=self.L
        X=self.__sampleFromDomain(num)      
        y=torch.norm(X-L,dim=1)     
        location=torch.where(y>self.r0)[0]
        X_out=X[location,:]      
        
        location=torch.where(y<=self.r0)[0]
        X_in=X[location,:]      

        return X_out,X_in


    def __sampleFromDomain(self,num):
        xmin,xmax,ymin,ymax=self.box
        x = torch.rand(num,device=self.device).view(-1,1) * (xmax - xmin) + xmin
        y = torch.rand(num,device=self.device).view(-1,1) * (ymax - ymin) + ymin
        X = torch.cat((x,y),dim=1)
        
        return X


    def SampleFromBoundary(self,num):
       
        xmin, xmax, ymin, ymax=self.box
        n=int(num/4)

        a = torch.rand(n).view(-1,1).to(self.device)*(xmax-xmin)+xmin
        b = torch.ones_like(a).to(self.device)*ymin
        P = torch.cat((a,b),dim=1)

        a = torch.rand(n).view(-1,1).to(self.device)*(xmax-xmin)+xmin
        b = torch.ones_like(a)*ymax
        P = torch.cat((P,torch.cat((a,b),dim=1)),dim=0)
        
        a = torch.rand(n).view(-1,1).to(self.device)*(ymax-ymin)+ymin
        b = torch.ones_like(a)*xmin   
        P = torch.cat((P,torch.cat((b,a),dim=1)),dim=0)
        
        a = torch.rand(n).view(-1,1).to(self.device)*(ymax-ymin)+ymin
        b = torch.ones_like(a)*xmax
        P = torch.cat((P,torch.cat((b,a),dim=1)),dim=0)

        return P.to(self.device)



# data=Data(r0=2/3,L=[0.,0.],box=[-1,1,1,-1],device='cpu')
# boundary=data.SampleFromBoundary(100)
# interface,f=data.SampleFromGamma(100)
# import matplotlib.pyplot as plt
# plt.scatter(interface[:,0],interface[:,1])
# plt.scatter(boundary[:,0],boundary[:,1])
# plt.show()


