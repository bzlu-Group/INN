# -*- coding: utf-8 -*-
import torch
import numpy as np
import os
pi=float(np.pi)

class Data(object):
    def __init__(self,box,device):
        self.box=torch.tensor(box).to(device)
        self.device=device

    def SampleFromGamma(self,num):
        """
        return: boundary points and related f_direction 
        """
        theta=2*pi*torch.rand(num,device=self.device).view(-1,1)            # [0,2pi]
        x=0.65*torch.cos(theta)**3
        y=0.65*torch.sin(theta)**3
        X=torch.cat((x,y),dim=1) 

        f1=0.65*3*torch.cos(theta)**2*(-torch.sin(theta))
        f2=0.65*3*torch.sin(theta)**2*torch.cos(theta)

        f_direction=torch.cat((f2,-f1),dim=1)
        f_direction=f_direction/torch.norm(f_direction,dim=1).view(-1,1) # 归一化
        return X.to(self.device),f_direction.to(self.device)
    

    def __r_interface(self,theta):
        x=(0.65*torch.cos(theta)**3).view(-1,1)
        y=(0.65*torch.sin(theta)**3).view(-1,1)
        X=torch.cat((x,y),dim=1) 

        return torch.norm(X,dim=1)



    def SampleFromDomain(self,num):
        X=self.__sampleFromDomain(num)  
        x=X[:,0]
        y=X[:,1]

        rr=np.cbrt(x.cpu().numpy())**2+np.cbrt(y.cpu().numpy())**2
        rr=torch.tensor(rr).to(self.device)  

        r=0.65**(2/3)
        
        location=torch.where(rr<r)[0]
        X_in=(X[location,:])
        location=torch.where(rr>r)[0]
        X_out=(X[location,:])

        return X_out.to(self.device),X_in.to(self.device)



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


