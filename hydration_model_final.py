!pip install pyDOE #Latin Hypercube Sampling

import torch
import torch.autograd as autograd         # computation of gradient
from torch import Tensor                  # tensor node in the computation graph
import torch.nn as nn                     # neural networks
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker
from sklearn.model_selection import train_test_split

import numpy as np
import time
from pyDOE import lhs         #Latin Hypercube Sampling
import scipy.io

#Set default dtype to float32
torch.set_default_dtype(torch.float)

#PyTorch random number generator
torch.manual_seed(1234)

# Random number generators in other libraries
np.random.seed(1234)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import scipy as sp
from scipy.integrate import odeint

#UNITS ARE IN MICRO METER
#Hydration model parameter: chnage based on your design requirements

rho = 2.349e-15 #2349 [kg/m^3] or 2.349e-15 [kg/μm^3]
C = 1.57e-16 #157[kg/m^3] or 1.57e-16 [kg/μm^3]  Moisture capacity
T_ref = 283.15 #[K]  Reference curing temperature: 10degC at t=0, 
w_c = 0.392 #water cemnet ration

T_initial=10 #degC  #Change initial curing temperature
T_init=T_initial+273.15 #K

#training parameter
steps=10000
lr=1e-3
layers = np.array([]) #layer_size = [2] + [32] * 3 + [1]
#Nu: Number of training points # Nf: Number of collocation points (Evaluate PDE)
N_u = 23 #Total number of data points for 'u'#N_u = 5(1% data), 23 (5% data), 46 (10% data), 100(20% data) training points.
N_f =10000 #Total number of collocation points 
data_points=N_u+N_f

class FCN(nn.Module):
    
    def __init__(self,layers):
        super().__init__() #call __init__ from parent class 
              
        'activation function'
        self.activation = nn.Tanh()

        'loss function'
        self.loss_function = nn.MSELoss(reduction ='mean')
    
        'Initialise neural network as a list using nn.Modulelist'  
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        
        self.iter = 0
    
        'Xavier Normal Initialization'
        for i in range(len(layers)-1):
            
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            
            # set biases to zero
            nn.init.zeros_(self.linears[i].bias.data)
            
    'foward pass'
    def forward(self,x):
        
        if torch.is_tensor(x) != True:         
            x = torch.from_numpy(x)                
        
        u_b = torch.from_numpy(ub).float().to(device)
        l_b = torch.from_numpy(lb).float().to(device)
                      
        #preprocessing input 
        x = (x - l_b)/(u_b - l_b) #feature scaling
        
        #convert to float
        a = x.float()
        
        for i in range(len(layers)-2):
            
            z = self.linears[i](a)
                        
            a = self.activation(z)
            
        a = self.linears[-1](a)
        
        return a
                        
    def loss_BC(self,x,y):
                
        loss_u = self.loss_function(self.forward(x), y)
                
        return loss_u
    
    def loss_PDE(self, X_train_Nf):
                        
        g = X_train_Nf.clone()
                        
        g.requires_grad = True
        
        u = self.forward(g)

        u1 = self.forward(1/g)
                
        u_x_t = autograd.grad(u,g,torch.ones([X_train_Nf.shape[0], 1]).to(device), retain_graph=True, create_graph=True)[0]
                                
        u_xx_tt = autograd.grad(u_x_t,g,torch.ones(X_train_Nf.shape).to(device), create_graph=True)[0]
                                                            
        u_x = u_x_t[:,[0]]
        
        u_t = u_x_t[:,[1]]
        
        u_xx = u_xx_tt[:,[0]]
                                        

        Cp = 9020*1  #9.02[J/g-degC] or 9020 [J/(kg.K)] heat capacity of concrete
        k = 9360*1*1e-6  #2.6 [W/(m*K)] 0r 2.6 [J/s(m*K)] or 2.6*3600 =9360 [J/hr(m*K)] or 9360*1e-9 [J/hr(nm*K)] Thermal conductivity of concrete
        Hu = 385000 #[J/kg] Total heat of hydration
        E_R = 5000 #[K]  Activation energy 
        alpha_u = 1.031*w_c/(0.194+w_c) #ultimate degree of hydration
        beta = 1.52 #hydration shape parameter
        t1 = 13

        def f(ode, t):
            #code starts here
	    #code to calculate: te w.r.to t (refer to the author's paper)
	    #code ends here
            return dtedt
     
        ode0=0.0001+np.zeros(data_points) #initial value of equivalent age       
        ode = odeint(f, ode0, t.reshape(180,))
        te=ode[:,0]
        
        te=torch.from_numpy(te).float().to(device)
        tau=1+te/t1

        f = (rho*Cp*u_t) - (k)*u_xx -Hu*C*torch.exp(E_R*((1/T_ref)-(u1)))* (alpha_u*beta/te)* (tau/te) ** beta * torch.exp(-(tau/te) ** beta)

        loss_f = self.loss_function(f,f_hat)
        return loss_f
    
    def loss(self,x,y,X_train_Nf):
        loss_u = self.loss_BC(x,y)
        loss_f = self.loss_PDE(X_train_Nf)
        loss_val = loss_u + loss_f
        return loss_val
     
    'callable for optimizer'                                       
    def closure(self):
        
        optimizer.zero_grad()
        
        loss = self.loss(X_train_Nu, U_train_Nu, X_train_Nf)
        
        loss.backward()
                
        self.iter += 1
        return loss


# Specify the full path to the .mat file
file_path = 'C:/Users/me/Documents/ASIF/Research/Cement Hydration/PINN_Hydration/1.Data/Data.mat' #Change path


# Load the .mat file from the specified file path
data = scipy.io.loadmat(file_path)

x = data['x']                                                  # 100 points between 0 and 100 [100x1]
t = data['t']                                                  # 180 time points between 0 and 180 [180x1] 
usol = (data['usol']+273.15)                                   # solution of 180x100 grid points
X, T = np.meshgrid(x,t)                         # makes 2 arrays X and T such that u(X[i],T[j])=usol[i][j] are a tuple

X_test = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))

# Domain bounds
lb = X_test[0]  
ub = X_test[-1] 
u_true = usol.flatten('F')[:,None] #Fortran style (Column Major)

'''Boundary Conditions'''

#Initial Condition -1 =< x =<1 and t = 0  
left_X = np.hstack((X[0,:][:,None], T[0,:][:,None])) #L1
left_U = usol[:,0][:,None]

#Boundary Condition x = -1 and 0 =< t =<1
bottom_X = np.hstack((X[:,0][:,None], T[:,0][:,None])) #L2
bottom_U = usol[-1,:][:,None]

#Boundary Condition x = 1 and 0 =< t =<1
top_X = np.hstack((X[:,-1][:,None], T[:,0][:,None])) #L3
top_U = usol[0,:][:,None]

X_train = np.vstack([left_X, bottom_X, top_X]) 
U_train = np.vstack([left_U, bottom_U, top_U]) 

#choose random N_u points for training
idx = np.random.choice(X_train.shape[0], N_u, replace=False) 

X_train_Nu = X_train[idx, :] #choose indices from  set 'idx' (x,t)
U_train_Nu = U_train[idx,:]      #choose corresponding u

U_train_Nu = U_train_Nu +(-T_ref+T_init) 
#U_train_Nu = U_train_Nu + 0.01*np.std(U_train_Nu)*np.random.randn(U_train_Nu.shape[0], U_train_Nu.shape[1])-T_ref+(T_init)

'''Collocation Points'''

# Latin Hypercube sampling for collocation points 
# N_f sets of tuples(x,t)
X_train_Nf = lb + (ub-lb)*lhs(2,N_f) 
X_train_Nf = np.vstack((X_train_Nf, X_train_Nu)) # append training points to collocation points


'Convert to tensor and send to GPU'
X_train_Nf = torch.from_numpy(X_train_Nf).float().to(device)
X_train_Nu = torch.from_numpy(X_train_Nu).float().to(device)
U_train_Nu = torch.from_numpy(U_train_Nu).float().to(device)
X_test = torch.from_numpy(X_test).float().to(device)
u = torch.from_numpy(u_true).float().to(device)
f_hat = torch.zeros(X_train_Nf.shape[0],1).to(device)

PINN = FCN(layers)
       
PINN.to(device)

'Neural Network Summary'
#print(PINN)

params = list(PINN.parameters())

'''Optimization'''

#optimizer = torch.optim.Adam(PINN.parameters(),lr=lr,amsgrad=False)

#'L-BFGS Optimizer'
optimizer = torch.optim.LBFGS(PINN.parameters(), lr, 
                              max_iter = steps, 
                              max_eval = None, 
                              tolerance_grad = 1e-11, 
                              tolerance_change = 1e-11, 
                              history_size = 100, 
                              line_search_fn = 'strong_wolfe')

start_time = time.time()

optimizer.step(PINN.closure)
    
    
elapsed = time.time() - start_time                
#print('Training time: %.2f' % (elapsed))

'test neural network'

u_pred = PINN(X_test)
error_vec = torch.linalg.norm((u-u_pred),2)/torch.linalg.norm(u,2)        # Relative L2 Norm of the error (Vector)  
u_pred = u_pred.cpu().detach().numpy()
u_pred = np.reshape(u_pred,(100,180),order='F')

#''' Model Accuracy ''' 
#print('Test Error: %.5f'  % (error_vec))

x1=X_test[:,0]
t1=X_test[:,1]

x1=x1.reshape(shape=X.shape).transpose(1,0).detach().cpu()
t1=t1.reshape(shape=X.shape).transpose(1,0).detach().cpu()
u_pred=u_pred - 273.15  #degC 
usol=usol - 273.15  #degC


"""Analysis/Post processing:"""

#Validation of average data:
def cal_average(num):
    sum_num = 0
    for t in num:
        sum_num = sum_num + t           

    avg = sum_num / len(num)
    return avg

#calculate avg of all temperature values over the domain for predicted
q=[]
s=[]
k=u_pred
kk=k.transpose()

j=t1.numpy()
jj=j.transpose()
for i in range(180):  
    p=cal_average(torch.from_numpy(kk)[i])
    q.append(p)

    r=cal_average(jj[i])
    s.append(r)

#calculate avg of all temperature values over the domain for solution
w=[]
k1=usol
kk1=k1.transpose()

for i in range(180):  
    v=cal_average(torch.from_numpy(kk1)[i])
    w.append(v)



# Plotting

# Plotting

#Ground truth
fig_1 = plt.figure(1, figsize=(12, 8))
plt.subplot(2, 2, 1)
cmap='Set1'
plt.pcolor(t1,x1, usol,
                  vmin=0., vmax=70., cmap='Set3') #Normalize within 0 and 70         
plt.colorbar()
plt.xlabel(r'$t(hours)$', fontsize=18)
plt.ylabel(r'$x(μm)$', fontsize=18)
plt.title('Ground Truth $T(x,t) ℃$', fontsize=18)

# Prediction
plt.subplot(2, 2, 2)
plt.pcolor(t1,x1, u_pred+1, vmin=0., vmax=70., cmap='Set3') #u_pred+1 used to show even temperature value in the plot
plt.colorbar()
plt.xlabel(r'$t (hours)$', fontsize=18)
plt.ylabel(r'$x (μm)$', fontsize=18)
plt.title('Predicted $\hat T(x,t)$ ℃', fontsize=18)

# Error
plt.subplot(2, 2, 3)
plt.pcolor(t1,x1, np.abs(usol - u_pred), cmap='Set3')
plt.colorbar()
plt.xlabel(r'$t (hours)$', fontsize=18)
plt.ylabel(r'$x (μm)$', fontsize=18)
plt.title(r'Absolute Error $|T(x,t)- \hat T(x,t)| ℃$', fontsize=18)
plt.tight_layout() 

#Validation
plt.subplot(2, 2, 4)
plt.plot(s,q,'*', color = 'blue', markersize = 2, label = 'Predicted Temperature (initial temperature =10degC)') #predicted
plt.plot(t,w,'o', color = 'green', markersize = 2, label = 'Exact Solution (initial temperature =10degC)') #solution

plt.xlabel(r'Time (hours)')
plt.ylabel(r'Temperature (degC)')
plt.title('Adiabatic temperature rise during cement hydration')#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.legend(loc='lower right')
#plt.axis('scaled')
plt.grid()
plt.show()
