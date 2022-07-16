# -*- coding: utf-8 -*-
import torch
import numpy as np

pi=float(np.pi)
torch.set_default_dtype(torch.float32)

class Data(object):

    def __init__(self,r0,L,box,device,t_space):
        self.r0=r0
        self.L=torch.tensor(L).to(device)
        self.box=torch.tensor(box).to(device)
        self.device=device
        self.t_space=t_space

    def SampleFromGamma(self,num):
        L=self.L
        t = torch.rand(num,device=self.device).view(-1,1) * (self.t_space[1]-self.t_space[0])+self.t_space[0] 
        psi=2*torch.rand(num,device=self.device).view(-1,1)   #[0,2pi]
        x=L[0]+(self.r0-0.25*torch.sin(t))*torch.cos(psi*pi)
        y=L[1]+(self.r0+0.5*t)*torch.sin(psi*pi)
        X=torch.cat((x,y,t),dim=1) 

        f_direction=torch.cat(((x-L[0])/(self.r0-0.25*torch.sin(t)),(y-L[1])/(self.r0+0.5*t)),dim=1)
        f_direction=f_direction/torch.norm(f_direction,dim=1).view(-1,1) #unit

        return X,f_direction
    

    def SampleFromDomain(self,num,initial=False):
        if initial:
            xmin,xmax,ymin,ymax=self.box
            x = torch.rand(num,device=self.device).view(-1,1) * (xmax - xmin) + xmin
            y = torch.rand(num,device=self.device).view(-1,1) * (ymax - ymin) + ymin
            t = torch.zeros_like(x)
            X = torch.cat((x,y,t),dim=1)  
            location=torch.where(((x/(self.r0))**2+(y/(self.r0))**2)>1)[0]
            X_out=X[location,:]                  
            location=torch.where(((x/(self.r0))**2+(y/(self.r0))**2)<1)[0]
            X_in=X[location,:]      
        else:
            X=self.__sampleFromDomain(num)      
            x,y,t=(X[:,[0,1]]-self.L).T[0],(X[:,[0,1]]-self.L).T[1],(X[2]).T[2]
            
            location=torch.where(((x/(self.r0-0.25*torch.sin(t)))**2+(y/(self.r0+0.5*t))**2)>1)[0]
            X_out=X[location,:]      
            
            location=torch.where(((x/(self.r0-0.25*torch.sin(t)))**2+(y/(self.r0+0.5*t))**2)<1)[0]
            X_in=X[location,:]      

        return X_out,X_in



    def __sampleFromDomain(self,num):

        xmin,xmax,ymin,ymax=self.box
        x = torch.rand(num,device=self.device).view(-1,1) * (xmax - xmin) + xmin
        y = torch.rand(num,device=self.device).view(-1,1) * (ymax - ymin) + ymin
        t = torch.rand(num,device=self.device).view(-1,1) * (self.t_space[1]-self.t_space[0])+self.t_space[0]
        X = torch.cat((x,y,t),dim=1)  

        return X



    def SampleFromBoundary(self,num):
       
        xmin, xmax, ymin, ymax=self.box
        n=int(num/4)

        a = torch.rand(n).view(-1,1).to(self.device)*(xmax-xmin)+xmin
        b = torch.ones_like(a,device=self.device).to(self.device)*ymin
        t = torch.rand(n).view(-1,1).to(self.device)*(self.t_space[1]-self.t_space[0])+self.t_space[0]
        P = torch.cat((a,b,t),dim=1)

        a = torch.rand(n).view(-1,1).to(self.device)*(xmax-xmin)+xmin
        b = torch.ones_like(a,device=self.device)*ymax
        t = torch.rand(n).view(-1,1).to(self.device)*(self.t_space[1]-self.t_space[0])+self.t_space[0]
        P = torch.cat((P,torch.cat((a,b,t),dim=1)),dim=0)
        
        a = torch.rand(n).view(-1,1).to(self.device)*(ymax-ymin)+ymin
        b = torch.ones_like(a,device=self.device)*xmin   
        t = torch.rand(n).view(-1,1).to(self.device)*(self.t_space[1]-self.t_space[0])+self.t_space[0]
        P = torch.cat((P,torch.cat((b,a,t),dim=1)),dim=0)
        
        a = torch.rand(n).view(-1,1).to(self.device)*(ymax-ymin)+ymin
        b = torch.ones_like(a,device=self.device)*xmax
        t = torch.rand(n).view(-1,1).to(self.device)*(self.t_space[1]-self.t_space[0])+self.t_space[0]
        P = torch.cat((P,torch.cat((b,a,t),dim=1)),dim=0)

        return P

