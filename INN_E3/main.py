import numpy as np 
import argparse
import torch
import time,os
import itertools
import random
import torch.optim as optim
from Tool import grad, data_transform, gradient, MGDA_train
from Net_type import DeepRitzNet
from GenerateData import Data

######### exact solution ###############
def u(x,label):

    x=x.t()  
    if label=='inner':
        u=(1/(1+10*(x[0]**2+x[1]**2))).view(-1,1)
    elif label=='out':
        u=(2/(1+10*(x[0]**2+x[1]**2))).view(-1,1)
    else:
        raise ValueError("invalid label for u(x)")
   
    return u


def f_grad(x,label):
    x=x.t()
    v=1+10*(x[0]**2+x[1]**2)
    uxx=(-2*10*v+2*(20*x[0])**2)/v**3
    uyy=(-2*10*v+2*(20*x[1])**2)/v**3

    return (-2*(uxx+uyy)).reshape(-1,1)

def inter_dirich(x):
    x=x.t()
    return (1/(1+10*(x[0]**2+x[1]**2))).view(-1,1)


def test_data_net(args,device):  
    
    step=0.02
    x = np.arange(-1, 1+step, step)
    y = np.arange(-1, 1+step, step)
    xx,yy=np.meshgrid(x,y)
    input_x=torch.tensor(xx).view(-1,1).to(device)
    input_y=torch.tensor(yy).view(-1,1).to(device)
    input=(torch.cat((input_x,input_y),1)).float()
    x=input[:,0]
    y=input[:,1]

    rr=np.cbrt(x.cpu().numpy())**2+np.cbrt(y.cpu().numpy())**2
    rr=torch.tensor(rr) 
    r=0.65**(2/3)
    location=torch.where(rr<r)[0]
    test_inner=(input[location,:])
    location=torch.where(rr>r)[0]
    test_out=(input[location,:])
    label_out=u(test_out,label='out')
    label_inner=u(test_inner,label='inner')

    return test_out.to(device),label_out.to(device),test_inner.to(device),label_inner.to(device)


def main(args):

    if torch.cuda.is_available and args.cuda:
        device='cuda'
        print('cuda is avaliable')
    else:
        device='cpu'
        
    ### test data
    test_out,label_out,test_inner,label_inner=test_data_net(args,device)
    
    ### train data
    data=Data(box=args.box,device=device)
    
    out,inner=data.SampleFromDomain(args.train_domian)
    out=out.T
    inner=inner.T
    gamma,f_direction=data.SampleFromGamma(args.train_gamma)
    gamma=gamma.T
    
    x_gamma,y_gamma,input_gamma=data_transform(gamma,device) 
    x_in,y_in,input_in=data_transform(inner,device)  
    x_out,y_out,input_out=data_transform(out,device) 
    
    input_boundary=data.SampleFromBoundary(args.train_boundary) 
    input_boundary_label=u(input_boundary,'out')  
    z=torch.ones(input_gamma.size()[0]).view(-1,1).to(device)

    print('out:',input_out.size())
    print('gamma',input_gamma.size())
    print('input_boundary',input_boundary.size())
    print('input_in',input_in.size())
    
    net_inner=DeepRitzNet(m=args.inner_unit).to(device) 
    net_out=DeepRitzNet(m=args.out_unit).to(device)     
    optimizer=optim.Adam(itertools.chain(net_inner.parameters(),net_out.parameters()),lr=args.lr)
    result=[]
    t0=time.time()
    task={}
    task_loss={}

    scale={}
    loss_history = []
    if not os.path.isdir('./outputs/'+args.filename+'/model'): os.makedirs('./outputs/'+args.filename+'/model')
    
    Traing_Mse_min=1e10
    Traing_Mse_min_epoch=0
    for epoch in range(args.nepochs):      
        optimizer.zero_grad()    
        U1=net_inner(input_in)
        U_1x,U_1y=gradient(U1,x_in,y_in,device)  
        U_1xx=grad(U_1x,x_in,device)
        U_1yy=grad(U_1y,y_in,device)
        loss_in=torch.mean((-2*(U_1xx+U_1yy)-f_grad(input_in,'inner' ))**2)
        
        U1_b=net_inner(input_gamma)
        U2_b_in=net_out(input_gamma)        
        loss_gammad=torch.mean((U2_b_in-U1_b-inter_dirich(input_gamma))**2)

        dU1_N=torch.autograd.grad(U1_b,input_gamma, grad_outputs=z, create_graph=True)[0]     
        dU2_N=torch.autograd.grad(U2_b_in,input_gamma, grad_outputs=z, create_graph=True)[0]
        G_NN=((dU2_N-2*dU1_N)*f_direction).sum(dim=1).view(-1,1)   
        loss_gamman=torch.mean((G_NN)**2)   
        
        U2 =net_out(input_out)
        U_2x,U_2y=gradient(U2,x_out,y_out,device)                
        U_2xx=grad(U_2x,x_out,device)
        U_2yy=grad(U_2y,y_out,device)       
        loss_out=torch.mean((-(U_2xx+U_2yy)-f_grad(input_out,'out' ))**2)
        loss_boundary=torch.mean((net_out(input_boundary)-input_boundary_label)**2)

        ## Extend MGD
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

        # ### resample training data 
        out,inner=data.SampleFromDomain(args.train_domian)
        out=out.T
        inner=inner.T
        gamma,f_direction=data.SampleFromGamma(args.train_gamma)
        gamma=gamma.T
           
        x_gamma,y_gamma,input_gamma=data_transform(gamma,device) 
        x_in,y_in,input_in=data_transform(inner,device)        
        x_out,y_out,input_out=data_transform(out,device)     
        
        input_boundary=data.SampleFromBoundary(args.train_boundary) 
        input_boundary_label=u(input_boundary,'out')  
        z=torch.ones(input_gamma.size()[0]).view(-1,1).to(device)

                
        if (epoch+1)%args.print_num==0:
            if  (epoch+1)%args.change_epoch==0 and optimizer.param_groups[0]['lr']>1e-6:
                optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']/2
             
            with torch.no_grad():    

                Mse_train=(loss_in+loss_out+loss_boundary+loss_gammad+loss_gamman).item()      
                print('Epoch,  Training MSE: ',epoch+1,Mse_train,optimizer.param_groups[0]['lr'])
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

    print('_______________________________________________________________________________')
    print('_______________________________________________________________________________')
    print('Training min MSE:' ,Traing_Mse_min)
    print('The epoch of the training min MSE:', Traing_Mse_min_epoch+1)
    pkl_in='outputs/'+args.filename+'/model/{}net_inner.pkl'.format(Traing_Mse_min_epoch)
    pkl_out='outputs/'+args.filename+'/model/{}net_out.pkl'.format(Traing_Mse_min_epoch)
    net_inner=torch.load(pkl_in)
    net_out=torch.load(pkl_out)
    # rela L_2
    L2_inner_loss=torch.sqrt(((net_inner(test_inner)-label_inner)**2).sum()/((label_inner)**2).sum())
    L2_out_loss=torch.sqrt(((net_out(test_out)-label_out)**2).sum()/((label_out)**2).sum())
    # L_infty
    L_inf_inner_loss=torch.max(torch.abs(net_inner(test_inner)-label_inner))
    L_inf_out_loss=torch.max(torch.abs(net_out(test_out)-label_out))
    print('L_infty:',max(L_inf_inner_loss.item(),L_inf_out_loss.item()))
    print('Rel. L_2:',(L2_inner_loss.item()*test_inner.size()[0]+L2_out_loss.item()*test_out.size()[0])/10201)
    print('Totle training time:',time.time()-t0)
        
if __name__ == '__main__':
    torch.cuda.set_device(0)
    parser = argparse.ArgumentParser() 
    parser.add_argument('--filename',type=str, default='results')
    parser.add_argument('--train_gamma', type=int, default=100)
    parser.add_argument('--train_domian', type=int, default=1000)   
    parser.add_argument('--train_boundary', type=int, default=100) 
    parser.add_argument('--inner_unit', type=int, default=80)
    parser.add_argument('--out_unit', type=int, default=80)   
    parser.add_argument('--print_num', type=int, default=100)
    parser.add_argument('--nepochs', type=int, default=25000)
    parser.add_argument('--lr', type=float, default=0.001)    
    parser.add_argument('--c', type=float, default=1)        
    parser.add_argument('--cuda', type=str, default=True)
    parser.add_argument('--box', type=list, default=[-1,1,-1,1])
    parser.add_argument('--change_epoch', type=int, default=2000)
    parser.add_argument('--save', type=str, default=False)
    args = parser.parse_args()
    main(args)

           
