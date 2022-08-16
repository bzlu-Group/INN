import numpy as np 
import argparse
import torch
import time,os
import itertools
import random
import torch.optim as optim
from Tool import grad,data_transform, gradient,MGDA_train
from Net_type import DeepRitzNet
import math
from GenerateData import Data

######### exact solution ###############
def u(x,label,a):
    x=x.t()
    if label=='inner':
        u=(a*((x[0]-1)**2+(x[1]-1)**2)-0.25*(a-1)+torch.exp(x[2])).view(-1,1)
    elif label=='out':
        u=((x[0]-1)**2+(x[1]-1)**2+2*torch.exp(-torch.cos(x[2]))).view(-1,1)
    else:
        raise ValueError("invalid label for u(x)")
   
    return u


def f_grad(x,label,a):   

    f=torch.ones(x.size()[0]).to(x.device)
    if label=='inner':  
        f=(-4*a*f).view(-1,1)
        
    elif label=='out':
        f=(-4*a*f).view(-1,1)   
    else:
        raise ValueError("invalid label for u(x)")
    
    return f

def phi_grad(x,f_direction,a,device): # Gammar
    xt=x.t()
    z=torch.ones(x.size()[0]).view(-1,1).to(device)
    du_out =torch.autograd.grad(u(x,'out',a),x, grad_outputs=z, create_graph=True)[0][:,[0,1]] 
    du_in =torch.autograd.grad(u(x,'inner',a),x, grad_outputs=z, create_graph=True)[0][:,[0,1]]
    dU = a*du_out-du_in
    f = dU*f_direction 
    p_grad=(f.sum(1)).view(-1,1)
  
    return p_grad
    

def test_data_net(args,t,device):  
 
    step=0.04
    x = np.arange(-1, 3+step, step)
    y = np.arange(-1, 3+step, step)
    L1=torch.tensor(args.L).to(device)
    xx,yy=np.meshgrid(x,y)
    input_x=torch.tensor(xx).view(-1,1).to(device)
    input_y=torch.tensor(yy).view(-1,1).to(device)
    tt=torch.ones_like(input_x,device=device)*t      # time
    input=(torch.cat((input_x,input_y,tt),1)).float()

    index_inner=torch.where((((input_x-L1[0])/(args.r0-0.25*torch.sin(t)))**2+((input_y-L1[1])/(args.r0+0.5*t))**2)<1)[0]
    index_out=torch.where((((input_x-L1[0])/(args.r0-0.25*torch.sin(t)))**2+((input_y-L1[1])/(args.r0+0.5*t))**2)>=1)[0]
    
    inner=input[index_inner,:]
    out=input[index_out,:]
    
    test_inner=inner.float().to(device).clone().detach()
    label_inner=u(test_inner,'inner',args.a).clone().detach()
    test_out=out.float().to(device).clone().detach()
    label_out=u(test_out,'out',args.a).clone().detach()

    return test_out,label_out,test_inner,label_inner

def main(args):
  
    if torch.cuda.is_available and args.cuda:
        device='cuda'
        print('cuda is avaliable')
    else:
        device='cpu'

    
    data=Data(r0=args.r0,L=args.L,box=args.box,device=device,t_space=args.t)
    # initial
    out_t0,inner_t0=data.SampleFromDomain(args.initial_data,initial=True)
    out_t0_lable=u(out_t0,'out',args.a)
    inner_t0_label=u(inner_t0,'inner',args.a)
    
    out,inner=data.SampleFromDomain(args.train_out)
    out=out.T
    inner=inner.T
    x_in,y_in,input_in=data_transform(inner,device)  
    x_out,y_out,input_out=data_transform(out,device) 
   
    gamma,f_direction=data.SampleFromGamma(args.train_gamma)
    gamma=gamma.T
    x_gamma,y_gamma,input_gamma=data_transform(gamma,device) 

    input_boundary=data.SampleFromBoundary(args.train_boundary) 
    input_boundary_label=u(input_boundary,'out',args.a)  
    g_D = u(input_gamma,'out',args.a)-u(input_gamma,'inner',args.a)
    z=torch.ones(g_D.size()).to(device)
    g_N = phi_grad(input_gamma,f_direction,args.a,device)
  
    print('out:',input_out.size())
    print('gamma',input_gamma.size())
    print('input_boundary',input_boundary.size())
    print('input_in',input_in.size())

    net_inner=DeepRitzNet(m=args.inner_unit).to(device) 
    net_out=DeepRitzNet(m=args.out_unit).to(device)
    optimizer=optim.Adam(itertools.chain(net_inner.parameters(),net_out.parameters()),lr=args.lr)

    t0=time.time()  
    task={}
    task_loss={}
    if not os.path.isdir('./outputs/'+args.filename+'/model'): os.makedirs('./outputs/'+args.filename+'/model')

    Traing_Mse_min=1e10
    Traing_Mse_min_epoch=0
    for epoch in range(args.nepochs):           
        optimizer.zero_grad()    
        U1=net_inner(input_in)
        U_1x,U_1y=gradient(U1,x_in,y_in,device)  
        U_1xx=grad(U_1x,x_in,device)
        U_1yy=grad(U_1y,y_in,device)
        loss_in=torch.mean((-(U_1xx+U_1yy)-f_grad(input_in,'inner',args.a))**2)       

        U1_b=net_inner(input_gamma)
        U2_b_in=net_out(input_gamma)     
        loss_gammad=torch.mean((U2_b_in-U1_b-g_D)**2)
       
        dU1_N=torch.autograd.grad(U1_b,input_gamma, grad_outputs=z, create_graph=True)[0][:,[0,1]]       
        dU2_N=torch.autograd.grad(U2_b_in,input_gamma, grad_outputs=z, create_graph=True)[0][:,[0,1]]       
        G_NN=((args.a*dU2_N-dU1_N)*f_direction).sum(dim=1).view(-1,1)   
        loss_gamman=torch.mean((G_NN-g_N)**2)   
        
        U2 =net_out(input_out) 
        U_2x,U_2y=gradient(U2,x_out,y_out,device)                  
        U_2xx=grad(U_2x,x_out,device)
        U_2yy=grad(U_2y,y_out,device)       
        loss_out=torch.mean((-(U_2xx+U_2yy)*args.a-f_grad(input_out,'out',args.a))**2)
        
        
        loss_boundary=torch.mean((net_out(input_boundary)-input_boundary_label)**2)
        loss_initial_in=torch.mean((net_inner(inner_t0)-inner_t0_label)**2)
        loss_initial_out=torch.mean((net_out(out_t0)-out_t0_lable)**2)


        # Extend MGDA
        NN=3 
        s=random.sample(range(1,NN),1)[0]
        task['bd_add_bn']=s/NN*loss_gammad/loss_gammad.data+(1-s/NN)*loss_gamman/loss_gamman.data
        task['loss_in_add_out']=loss_in+loss_out
        task['initial']=loss_initial_in+loss_initial_out
        task['outb']=loss_boundary


        task_loss['1']=loss_in.item()
        task_loss['2']=loss_gammad.item()
        task_loss['3']=loss_gamman.item()
        task_loss['4']=loss_out.item()
        task_loss['5']=loss_boundary.item()    
        task_loss['6']=loss_initial_in.item()   
        task_loss['7']=loss_initial_out.item()  
 
        scale=MGDA_train(epoch,task,task_loss,net_inner,net_out,optimizer,device,s,NN)
        loss=(loss_in*scale['loss_in']+scale['loss_gammad']*loss_gammad+loss_gamman*scale['loss_gamman']
        +scale['loss_out']*loss_out+scale['loss_boundary']*loss_boundary
        +scale['loss_initial_in']*loss_initial_in+scale['loss_initial_out']*loss_initial_out)
        loss.backward(retain_graph=True)
        optimizer.step()

        ########### resample training data #############
        out_t0,inner_t0=data.SampleFromDomain(args.initial_data,initial=True)
        out_t0_lable=u(out_t0,'out',args.a)
        inner_t0_label=u(inner_t0,'inner',args.a)

        out,inner=data.SampleFromDomain(args.train_out)
        out=out.T
        inner=inner.T
        x_in,y_in,input_in=data_transform(inner,device) 
        x_out,y_out,input_out=data_transform(out,device) 
    
        gamma,f_direction=data.SampleFromGamma(args.train_gamma)
        gamma=gamma.T
        x_gamma,y_gamma,input_gamma=data_transform(gamma,device) 

        input_boundary=data.SampleFromBoundary(args.train_boundary) 
        input_boundary_label=u(input_boundary,'out',args.a)  
        g_D = u(input_gamma,'out',args.a)-u(input_gamma,'inner',args.a)
        z=torch.ones(g_D.size()).to(device)
        g_N = phi_grad(input_gamma,f_direction,args.a,device)
             
        if (epoch+1)%args.print_num==0:

            if  (epoch+1)%args.change_epoch==0 and optimizer.param_groups[0]['lr']>1e-6:
                optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']/2
                      
            with torch.no_grad():

                print('Epoch : ',epoch+1)          
                Mse_train=(loss_in+loss_out+loss_boundary+loss_gammad+loss_gamman+loss_initial_in+loss_initial_out).item()      
                print('Training MSE',Mse_train)
                print('Gamma_d,Gamma_n,Omega1,Omega2,partial_Omega,initial',loss_gammad.item(),loss_gamman.item(),
                loss_in.item(),loss_out.item(),loss_boundary.item(),(loss_initial_in+loss_initial_out).item())

                if epoch>args.nepochs*0.95:             
                    torch.save(net_inner, 'outputs/'+args.filename+'/model/{}net_inner.pkl'.format(epoch))
                    torch.save(net_out, 'outputs/'+args.filename+'/model/{}net_out.pkl'.format(epoch))
                    if Traing_Mse_min>Mse_train:
                        Traing_Mse_min=Mse_train
                        Traing_Mse_min_epoch=epoch       


    print('_______________________________________________________________________________')
    print('_______________________________________________________________________________')
    print('Training min MSE:' ,Traing_Mse_min)
    print('The epoch of the training min MSE:', Traing_Mse_min_epoch+1)
    pkl_in='outputs/'+args.filename+'/model/{}net_inner.pkl'.format(Traing_Mse_min_epoch)
    pkl_out='outputs/'+args.filename+'/model/{}net_out.pkl'.format(Traing_Mse_min_epoch)
    net_inner=torch.load(pkl_in)
    net_out=torch.load(pkl_out)

    for i in range(1,5): # t=0.1,0.2,0.3,0.4
        TT=torch.tensor(i/10)
        test_out,label_out,test_inner,label_inner=test_data_net(args,TT,device)
        # L_infty            
        lrele_inner_loss=torch.max(torch.abs(net_inner(test_inner)-label_inner))
        lrele_out_loss=torch.max(torch.abs(net_out(test_out)-label_out))
        # L2
        L2_inner_loss=torch.sqrt(((net_inner(test_inner)-label_inner)**2).sum()/((label_inner)**2).sum())
        L2_out_loss=torch.sqrt(((net_out(test_out)-label_out)**2).sum()/((label_out)**2).sum())
        rele_loss=(L2_inner_loss.item()*test_inner.size()[0]+L2_out_loss.item()*test_out.size()[0])/10201

        print(epoch+1,TT)
        print('L_infty: ',max(lrele_inner_loss.item(),lrele_out_loss.item()))
        print('Rel. L_2: ',rele_loss)
        print('-'*50)

    print('totle use time:',time.time()-t0)
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser() 
    parser.add_argument('--filename',type=str, default='output')
    parser.add_argument('--train_gamma', type=int, default=100)
    parser.add_argument('--train_out', type=int, default=2000)   
    parser.add_argument('--train_boundary', type=int, default=800) 
    parser.add_argument('--initial_data', type=int, default=400)
    parser.add_argument('--inner_unit', type=int, default=320)
    parser.add_argument('--out_unit', type=int, default=320)    
    parser.add_argument('--print_num', type=int, default=100)
    parser.add_argument('--nepochs', type=int, default=60000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--cuda', type=str, default=True)
    parser.add_argument('--r0', type=float, default=0.5)
    parser.add_argument('--t', type=float, default=[0,0.5])
    parser.add_argument('--a', type=float, default=20)
    parser.add_argument('--L', type=list, default=np.array([1,1]))
    parser.add_argument('--box', type=list, default=np.array([-1,3,-1,3]))
    parser.add_argument('--change_epoch', type=int, default=4000)
    args = parser.parse_args()
    main(args)

        
           
