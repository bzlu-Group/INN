# -*- coding: utf-8 -*-
import torch
import numpy as np

def star_r(theta, theta_i, case):
    R=6/7
    theta_t = pi/5
    theta_r = pi/7
    if case==1:
        r=R*np.sin(theta_t/2)/np.sin(theta_t/2+theta-theta_r-2*(theta_i-1)*pi/5)
        
    elif case==2:
        r=R*np.sin(theta_t/2)/np.sin(theta_t/2-theta+theta_r+2*(theta_i-1)*pi/5)

    return r

def deal_interface_data(x,y,tol=0.03):
    """
    deal with the interface data
    """
    k=1
    for i,x_i in enumerate(x):
        index=np.where(np.sqrt((x-x_i)**2+(y-y[i])**2)<tol)[0]
        index=np.sort(index)[1:]
        
        if len(index)>0:
            k=0
            x_new=np.delete(x,index)
            y_new=np.delete(y,index)
            break
    if k==1:

        return x,y
    else:
        return deal_interface_data(x_new, y_new,tol)


pi=float(np.pi)
class Data(object):
    def __init__(self,box,device):
        self.box=torch.tensor(box).to(device)
        self.device=device

    def __interface_fdirection(self,x,y):
        k=torch.tensor([0.9560978966113615, 0.13545920388352728, -2.0765213965723377, 5.510454187211495, -0.5381229467220272])
        b=torch.tensor([-0.36645488605394094, 0.2672907544694173, 0.610467011284964, 1.4834023058193493, -0.3007870241014712])
        norm=torch.tensor([[ 0.72279486, -0.69106265],[-0.89421925,  0.44762923],[0.43388374, 0.90096887],[ 0.17855689, -0.98392959],[-0.88059553, -0.47386866]])
        f_direction=[]
        
        for i,x_i in enumerate(x):
            kk=None
            for j in range(5):
                if torch.abs(k[j]*x_i+b[j]-y[i])<1e-6:
                    if kk==None:
                        kk=norm[j]
                    else:
                        kk+=norm[j]
                else:
                    pass
            f_direction.append((kk/torch.norm(kk)).numpy())
        return torch.tensor(f_direction).float()

    def SampleFromGamma(self,num):
        R=6/7
        theta_t = pi/5
        theta_r = pi/7
        theta_set=np.linspace(-2/35*pi,68/35*pi,num,endpoint=False)

        r_interface=[]
        theta_i_list=[]
        for i, theta in enumerate(theta_set):
            for theta_i in range(1,6):
                if (theta_r+(2*theta_i-2)*pi/5) <= theta and theta < (theta_r+(2*theta_i-1)*pi/5):
                    r=star_r(theta,theta_i,case=1)
                    kk=theta_i
                elif (theta_r+(2*theta_i-3)*pi/5) <= theta and theta < (theta_r+(2*theta_i-2)*pi/5):
                    kk=theta_i
                    r=star_r(theta,theta_i,case=2)

            theta_i_list.append(kk)
            r_interface.append(r)
        
        r_interface=np.array(r_interface)
        x=r_interface*np.cos(theta_set)
        y=r_interface*np.sin(theta_set)
        x,y=deal_interface_data(x,y)

        x=torch.tensor(x).view(-1,1)
        y=torch.tensor(y).view(-1,1)
        X=torch.cat((x,y),dim=1).float() 
        f_direction=self.__interface_fdirection(x,y)
        
        return X.to(self.device) , f_direction.to(self.device)
    
    def __inner_or_out(self,X):

        R=6/7
        theta_t = pi/5
        theta_r = pi/7
        x=X[:,0]
        y=X[:,1]
        
        rr=torch.norm(X,dim=1)
        theta_set=-1000*torch.ones_like(x)
        # I
        index_x=torch.where(x>=0)[0]
        index_y=torch.where(y>=0)[0]
        loca=list(set(index_x.numpy())&set(index_y.numpy()))
        theta_set[loca]=torch.arccos(x[loca]/rr[loca])
        # II
        index_x=torch.where(x<=0)[0]
        index_y=torch.where(y>0)[0]
        loca=list(set(index_x.numpy())&set(index_y.numpy()))
        theta_set[loca]=pi-torch.arccos(-x[loca]/rr[loca]) 
        # III
        index_x=torch.where(x<=0)[0]
        index_y=torch.where(y<0)[0]
        loca=list(set(index_x.numpy())&set(index_y.numpy()))
        theta_set[loca]=pi+torch.arccos(-x[loca]/rr[loca])   
        # IIII
        index_x=torch.where(x>=0)[0]
        index_y=torch.where(y<0)[0]
        loca=list(set(index_x.numpy())&set(index_y.numpy()))
        theta_set[loca]=2*pi-torch.arccos(x[loca]/rr[loca])              
        
        r_interface=[]
        for i, theta in enumerate(theta_set):
            for theta_i in range(1,6):
                if (theta_r+(2*theta_i-2)*pi/5) <= theta and theta < (theta_r+(2*theta_i-1)*pi/5):
                    r=star_r(theta,theta_i,case=1)
                elif (theta_r+(2*theta_i-3)*pi/5) <= theta and theta < (theta_r+(2*theta_i-2)*pi/5):
                    r=star_r(theta,theta_i,case=2)
                elif (theta_r+(2*theta_i-2)*pi/5)+2*pi <= theta and theta < (theta_r+(2*theta_i-1)*pi/5)+2*pi:
                    r=star_r(theta,theta_i,case=1)
                elif  (theta_r+(2*theta_i-3)*pi/5)+2*pi <= theta and theta < (theta_r+(2*theta_i-2)*pi/5)+2*pi:
                    r=star_r(theta,theta_i,case=2)
            r_interface.append(r)   

        r_interface=torch.tensor(r_interface)
        index=torch.where(rr<r_interface)[0]
        X_in=X[index]
        index=torch.where(rr>r_interface)[0]
        X_out=X[index]

        return X_out,X_in


    def SampleFromDomain(self,num):
        
       
        X=self.__sampleFromDomain(num)      
        X_out,X_in =self.__inner_or_out(X)

        return X_out.to(self.device),X_in.to(self.device)


    def __sampleFromDomain(self,num):
        xmin,xmax,ymin,ymax=self.box.cpu()
        x=torch.linspace(xmin,xmax,num)[1:-1]
        y=torch.linspace(xmin,xmax,num)[1:-1]
        x,y=torch.meshgrid(x,y)

        x=x.reshape(-1,1)
        y=y.reshape(-1,1)
        X = torch.cat((x,y),dim=1)

        return X


    def SampleFromBoundary(self,num):
       
        xmin, xmax, ymin, ymax=self.box
        n=int(num/4)

        # left
        a = torch.linspace(xmin,xmax,num).to(self.device).view(-1,1)
        b = torch.ones_like(a).to(self.device)*ymin
        P = torch.cat((a,b),dim=1)
        
        # right
        a = torch.linspace(xmin,xmax,num).to(self.device).view(-1,1)
        b = torch.ones_like(a)*ymax
        P = torch.cat((P,torch.cat((a,b),dim=1)),dim=0)
        
        a = torch.linspace(ymin,ymax,num).to(self.device).view(-1,1)[1:-1]
        b = torch.ones_like(a)*xmin   
        P = torch.cat((P,torch.cat((b,a),dim=1)),dim=0)
        
        a = torch.linspace(ymin,ymax,num).to(self.device).view(-1,1)[1:-1]
        b = torch.ones_like(a)*xmax
        P = torch.cat((P,torch.cat((b,a),dim=1)),dim=0)

        return P.to(self.device)

