import numpy as np 
import argparse
import torch
import time,os
import itertools
import random
import torch.optim as optim
from Tool import grad, data_transform, gradient, MGDA_train
from Net_Fourier import DeepRitzNet
import torch.nn as nn
from GenerateData import Data


################## exact solution ##################
def u(x,label):
    x=x.T  
    if label=='inner':
        u=(x[0]**2+x[1]**2).view(-1,1) 
    elif label=='out':
        u=(torch.sin(10*x[0])*torch.cos(10*x[1])).view(-1,1) 
    else:
        raise ValueError("invalid label for u(x)")
   
    return u

    
def f_grad(x,label,device):
    xt=x.t() 
    if label=='inner':  
        f = 40.+100*u(x,label)
    elif label=='out':
        f = 800*u(x,label)     
    else:
        raise ValueError("invalid label for u(x)")
    
    return f


def phi_grad(x,f_direction,device): 
    xt=x.t()
    z=torch.ones(x.size()[0]).view(-1,1).to(device)
    
    du_out =torch.autograd.grad(u(x,'out'),x, grad_outputs=z, create_graph=True)[0]*(-1)
    du_in =torch.autograd.grad(u(x,'inner'),x, grad_outputs=z, create_graph=True)[0]*(-10)
    dU=du_out-du_in
  
    f = dU*f_direction 
    p_grad=(f.sum(1)).view(-1,1)

    return p_grad

def main(args):

    if torch.cuda.is_available and args.cuda:
        device='cuda'
        print('cuda is avaliable')
    else:
        device='cpu'
        
    center=torch.tensor(args.L).to(device)
    r0=args.r0
  
    data=Data(r0=args.r0,L=args.L,box=args.box,device=device)
    out,inner=data.SampleFromDomain(args.train_out)
    out=out.T
    inner=inner.T     
    x_in,y_in,input_in=data_transform(inner,device)  
    x_out,y_out,input_out=data_transform(out,device) 
    
    gamma,f_direction=data.SampleFromGamma(args.train_gamma)
    gamma=gamma.T
    x_gamma,y_gamma,input_gamma=data_transform(gamma,device) 
    input_boundary=data.SampleFromBoundary(args.train_boundary) 
    input_boundary_label=u(input_boundary,'out')  
    g_D = u(input_gamma,'out')-u(input_gamma,'inner')
    z=torch.ones(g_D.size()).to(device)
    g_N = phi_grad(input_gamma,f_direction,device)
    
    print('out:',input_out.size())
    print('gamma',input_gamma.size())
    print('input_boundary',input_boundary.size())
    print('input_in',input_in.size())


    net_inner=DeepRitzNet(m=args.inner_unit,device=device).to(device) 
    net_out=DeepRitzNet(m=args.out_unit,device=device).to(device) 
    
    for name, param in net_inner.named_parameters():
        if "FF" in name:
            param.requires_grad = False
    for name, param in net_out.named_parameters():
        if "FF" in name:
            param.requires_grad = False



    optimizer=optim.Adam(itertools.chain(net_inner.parameters(),net_out.parameters()),lr=args.lr)
    result=[]
    t0=time.time()
    task={}
    task_loss={}

    train_loss=[]
    test_loss=[]
    scale_record=[]
    loss_history = []
    test_record=[]
    if not os.path.isdir('./outputs/'+args.filename+'/model'): os.makedirs('./outputs/'+args.filename+'/model')
    
    Mse_train_f = 1e-5
    Traing_Mse_min=1e10
    Traing_Mse_min_epoch=0
    for epoch in range(args.nepochs):
        optimizer.zero_grad()    
        U1=net_inner(input_in)
        U_1x,U_1y=gradient(U1,x_in,y_in,device)  
        U_1xx=grad(U_1x,x_in,device)
        U_1yy=grad(U_1y,y_in,device)
        loss_in=torch.mean((10*(U_1xx+U_1yy)+100*U1-f_grad(input_in,'inner',device))**2) ############
        
        U1_b=net_inner(input_gamma)
        U2_b_in=net_out(input_gamma)      
        loss_gammad=torch.mean((U2_b_in-U1_b-g_D)**2)
       
        dU1_N=torch.autograd.grad(U1_b,input_gamma, grad_outputs=z, create_graph=True)[0] 
        dU2_N=torch.autograd.grad(U2_b_in,input_gamma, grad_outputs=z, create_graph=True)[0]     
        G_NN=((-1*dU2_N+10*dU1_N)*f_direction).sum(dim=1).view(-1,1)   
        loss_gamman=torch.mean((G_NN-g_N)**2)   
        
        U2 =net_out(input_out) 
        U_2x,U_2y=gradient(U2,x_out,y_out,device)       
        U_2xx=grad(U_2x,x_out,device)
        U_2yy=grad(U_2y,y_out,device)       
        loss_out=torch.mean(((U_2xx+U_2yy)+1000*U2-f_grad(input_out,'out',device))**2)
        loss_boundary=torch.mean((net_out(input_boundary)-input_boundary_label)**2)

        # Extend MGD
        NN=10 
        s=random.sample(range(1,NN),1)[0]
        task['bd_add_bn']=s/NN*loss_gammad/loss_gammad.data+(1-s/NN)*loss_gamman/loss_gamman.data
        task['loss_in_add_out']=loss_in+loss_out
        task['outb']=loss_boundary       

        task_loss['1']=loss_in.item()
        task_loss['2']=loss_gammad.item()
        task_loss['3']=loss_gamman.item()
        task_loss['4']=loss_out.item()
        task_loss['5']=loss_boundary.item()   

        scale=MGDA_train(epoch,task,task_loss,net_inner,net_out,optimizer,device,s,NN,c=args.c)
        loss=(loss_in*scale['loss_in']+scale['loss_gammad']*loss_gammad
        +loss_gamman*scale['loss_gamman']+scale['loss_out']*loss_out
            +scale['loss_boundary']*loss_boundary)
        loss.backward(retain_graph=True)
        optimizer.step()


        # ### resample training data points
        # out,inner=data.SampleFromDomain(args.train_out)
        # out=out.T
        # inner=inner.T     
        # x_in,y_in,input_in=data_transform(inner,device)  
        # x_out,y_out,input_out=data_transform(out,device) 
        
        # gamma,f_direction=data.SampleFromGamma(args.train_gamma)
        # gamma=gamma.T
        # x_gamma,y_gamma,input_gamma=data_transform(gamma,device) 

        # input_boundary=data.SampleFromBoundary(args.train_boundary) 
        # input_boundary_label=u(input_boundary,'out')  
        # g_D = u(input_gamma,'out')-u(input_gamma,'inner')
        # z=torch.ones(g_D.size()).to(device)
        # g_N = phi_grad(input_gamma,f_direction,device)

    
        if (epoch+1)%args.print_num==0:          
            if  (epoch+1)%args.change_epoch==0 and optimizer.param_groups[0]['lr']>1e-6:  
                optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.5
         
            with torch.no_grad():    

                Mse_train=(loss_in+loss_out+loss_boundary+loss_gammad+loss_gamman).item()      
                print('Epoch,  Training MSE: ',epoch+1,Mse_train,optimizer.param_groups[0]['lr'])
                print('Gamma_d,Gamma_n,Omega1,Omega2,partial_Omega: ',loss_gammad.item(),loss_gamman.item(),loss_in.item(),loss_out.item(),loss_boundary.item())
                print('*****************************************************')  


                if epoch>args.nepochs*0.98:             
                    torch.save(net_inner, 'outputs/'+args.filename+'/model/{}net_inner.pkl'.format(epoch))
                    torch.save(net_out, 'outputs/'+args.filename+'/model/{}net_out.pkl'.format(epoch))
                    if Traing_Mse_min>Mse_train:
                        Traing_Mse_min=Mse_train
                        Traing_Mse_min_epoch=epoch

                if args.save:
                    loss_history.append([epoch,loss_in.item(),loss_gammad.item(),loss_out.item(),loss_boundary.item(),loss_gamman.item()])
                    loss_record = np.array(loss_history)
                    np.savetxt('outputs/'+args.filename+'/loss_record.txt', loss_record)           

            if abs(Mse_train_f-Mse_train)/Mse_train_f>1e-6 or Mse_train>100:
                Mse_train_f=Mse_train
            else:
                torch.save(net_inner, 'outputs/'+args.filename+'/model/{}net_inner.pkl'.format(epoch))
                torch.save(net_out, 'outputs/'+args.filename+'/model/{}net_out.pkl'.format(epoch))
                if Traing_Mse_min>Mse_train:
                    Traing_Mse_min=Mse_train
                    Traing_Mse_min_epoch=epoch                
                print('Stop training, tolerance of Mse_train < 1e-6')
                break 

    print('_______________________________________________________________________________')
    print('_______________________________________________________________________________')
    print('Training min MSE:' ,Traing_Mse_min)
    print('The epoch of the training min MSE:', Traing_Mse_min_epoch+1)
    pkl_in='outputs/'+args.filename+'/model/{}net_inner.pkl'.format(Traing_Mse_min_epoch)
    pkl_out='outputs/'+args.filename+'/model/{}net_out.pkl'.format(Traing_Mse_min_epoch)
    net_inner=torch.load(pkl_in)
    net_out=torch.load(pkl_out)

    # L_2
    L2_loss=((net_out(input_out)-u(input_out,'out'))**2).sum()+((net_inner(input_in)-u(input_in,'inner'))**2).sum()+((net_out(input_boundary)-u(input_boundary,'out'))**2).sum()
    L2_loss=torch.sqrt(L2_loss/(input_in.size()[0]+input_out.size()[0]+input_boundary.size()[0]))

    print('L_2:',L2_loss)
    print('totle use time:',time.time()-t0)
        
        
if __name__ == '__main__':
    torch.cuda.set_device(1) 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--filename',type=str, default='result')
    parser.add_argument('--train_gamma', type=int, default=200)           # 200/1000/400 
    parser.add_argument('--train_out', type=int, default=2600)            # 200/2600/800 
    parser.add_argument('--train_boundary', type=int, default=800) 
    parser.add_argument('--inner_unit', type=int, default=160)
    parser.add_argument('--out_unit', type=int, default=160)
    parser.add_argument('--save', type=str, default=False)
    parser.add_argument('--c', type=int, default=1e2)
    parser.add_argument('--print_num', type=int, default=100)
    parser.add_argument('--nepochs', type=int, default=25000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--cuda', type=str, default=True)
    parser.add_argument('--r0', type=float, default=0.5)
    parser.add_argument('--L', type=list, default=[0.,0.])
    parser.add_argument('--box', type=list, default=[-1,1,-1,1])
    parser.add_argument('--change_epoch', type=int, default=2000)

    args = parser.parse_args()
    main(args)

        
           
