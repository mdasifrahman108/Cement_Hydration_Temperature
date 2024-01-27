"""Hydration Model_Final.py


! pip install pyDOE #Latin Hypercube Sampling

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

print(device)

if device == 'cuda':
    print(torch.cuda.get_device_name())

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
layers = np.array([2,24,24,24,1]) #layer_size = [2] + [24] * 3 + [1]
#Nu: Number of training points # Nf: Number of collocation points (Evaluate PDE)
N_u = 23 #Total number of data points for 'u'#N_u = 5(1% data), 23 (5% data), 46 (10% data), 100(20% data) training points.
N_f =10000 #Total number of collocation points
data_points=N_u+N_f

def plot3D(x,t,y):
  fig_1 = plt.figure(1, figsize=(8, 5))
  x_plot =x.squeeze(1)
  t_plot =t.squeeze(1)
  X,T= torch.meshgrid(x_plot,t_plot)
  F_xt = y
  ax = plt.axes(projection='3d')
  ax.plot_surface(T.numpy(), X.numpy(), F_xt.numpy(),cmap="jet") #rainbow
  ax.set_xlabel('t')
  ax.set_ylabel('x')
  #plt.title(r'Solution $u(t,x)$', fontsize=15)
  plt.show()

def plot3D_Matrix(x,t,y):
  fig_1 = plt.figure(1, figsize=(8, 5))
  X,T= x,t
  F_xt = y
  ax = plt.axes(projection='3d')
  ax.plot_surface(T.numpy(), X.numpy(), F_xt.numpy(),cmap="jet") #rainbow
  ax.set_xlabel('t')
  ax.set_ylabel('x')
  #plt.title(r'Prediction $\hat u(t,x)$', fontsize=15)
  plt.show()

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
            T=u1
            T=T.detach().numpy()
            T = T.reshape(data_points,)  #Odeint can take only 1d array
            T = torch.from_numpy(T).float().to(device)
            te=str(ode)[0]
            dtedt = (torch.exp(E_R*((1/T_ref)-(T))))
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

        if self.iter % 100 == 0:

            print(loss)
        #pq=loss.detach().numpy()
        #plt.plot(pq)

        return loss

from google.colab import drive
import os
drive.mount('/content/drive')

data = scipy.io.loadmat('/content/drive/MyDrive/Colab Notebooks/Data.mat')
x = data['x']                                                  # 100 points between 0 and 100 [100x1]
t = data['t']                                                  # 180 time points between 0 and 180 [180x1]
usol = (data['usol']+273.15)                  # solution of 180x100 grid points:
X, T = np.meshgrid(x,t)                         # makes 2 arrays X and T such that u(X[i],T[j])=usol[i][j] are a tuple

print(x.shape,t.shape,usol.shape)
print(X.shape,T.shape)

X_test = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))

# Domain bounds
lb = X_test[0]
ub = X_test[-1]
u_true = usol.flatten('F')[:,None] #Fortran style (Column Major)
print(lb,ub)

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

print("Original shapes for X and U:",X.shape,usol.shape)
print("Boundary shapes for the edges:",left_X.shape,bottom_X.shape,top_X.shape)
print("Available training data:",X_train.shape,U_train.shape)
print("Final training data:",X_train_Nu.shape,U_train_Nu.shape)
print("Total collocation points:",X_train_Nf.shape)
print("We have",X_train.shape[0],"points at the boundary. We will select",X_train_Nu.shape[0],"points to train our model.")

#Plot collocation point
fig_1 = plt.figure(1, figsize=(30, 5))
plt.plot(X_train_Nu[:,0],X_train_Nu[:,1], '*', color = 'red', markersize = 5, label = 'Boundary collocation points= 100')
plt.plot(X_train_Nf[:,0],X_train_Nf[:,1], 'o', markersize = 0.5, label = 'PDE collocation points = 10,100')

plt.xlabel(r'$x(μm)$', fontsize=18)
plt.ylabel(r'$t(hours)$', fontsize=18)
plt.title('Collocation points', fontsize=18)
plt.axis('scaled')
plt.show()
plt.tight_layout()
# fig.savefig('collocation_points_Helmholtz.jpg', dpi = 500)

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
print(PINN)

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
print('Training time: %.2f' % (elapsed))

import torch

# Define the path to save the model
model_path = 'PINN.pt'

# Save the trained model
torch.save(PINN.state_dict(), model_path)

from google.colab import files
# Download the saved model
#files.download(model_path)

#Remove ''' and use the following code to load the trained, saved, and downloaded model for prediction

'''
# Define the path to the downloaded model on your local machine
downloaded_model_path = '/content/drive/MyDrive/Colab Notebooks/Completed/PINN.pt'

# Load the downloaded model
downloaded_model_state_dict = torch.load(downloaded_model_path)

# Create the model architecture
PINN = FCN(layers)

# Load the downloaded parameters into the model
PINN.load_state_dict(downloaded_model_state_dict)

# Move the model to the device
PINN.to(device)

# Switch the model to evaluation mode
PINN.eval()'''

'test neural network'

u_pred = PINN(X_test)
error_vec = torch.linalg.norm((u-u_pred),2)/torch.linalg.norm(u,2)        # Relative L2 Norm of the error (Vector)
u_pred = u_pred.cpu().detach().numpy()
u_pred = np.reshape(u_pred,(100,180),order='F')

''' Model Accuracy '''
#error_vec, u_pred = PINN.test()
print('Test Error: %.5f'  % (error_vec))

x1=X_test[:,0]
t1=X_test[:,1]

x1=x1.reshape(shape=X.shape).transpose(1,0).detach().cpu()
t1=t1.reshape(shape=X.shape).transpose(1,0).detach().cpu()
u_pred=u_pred - 273.15  #degC
usol=usol - 273.15  #degC

# Plotting

#Ground truth
fig_1 = plt.figure(1, figsize=(18, 5))
plt.subplot(1, 3, 1)
cmap='Set1'
plt.pcolor(t1,x1, usol,
                  vmin=0., vmax=70., cmap='Set3') #Normalize within 0 and 70
plt.colorbar()
plt.xlabel(r'$t(hours)$', fontsize=18)
plt.ylabel(r'$x(μm)$', fontsize=18)
plt.title('Ground Truth $T(x,t) ℃$', fontsize=18)

# Prediction
plt.subplot(1, 3, 2)
plt.pcolor(t1,x1, u_pred+1, vmin=0., vmax=70., cmap='Set3') #u_pred+1 used to show even temperature value in the plot
plt.colorbar()
plt.xlabel(r'$t (hours)$', fontsize=18)
plt.ylabel(r'$x (μm)$', fontsize=18)
plt.title('Predicted $\hat T(x,t)$ ℃', fontsize=18)

# Error
plt.subplot(1, 3, 3)
plt.pcolor(t1,x1, np.abs(usol - u_pred), cmap='Set3')
plt.colorbar()
plt.xlabel(r'$t (hours)$', fontsize=18)
plt.ylabel(r'$x (μm)$', fontsize=18)
plt.title(r'Absolute Error $|T(x,t)- \hat T(x,t)| ℃$', fontsize=18)
plt.tight_layout()

from google.colab import files
plt.savefig("a.jpg")
files.download("a.jpg")

plot3D_Matrix(x1,t1,torch.from_numpy(u_pred)) #prediction

plot3D(torch.from_numpy(x),torch.from_numpy(t),torch.from_numpy(usol)) #solution

"""Analysis/Post processing:"""

#Analysis for the box plot
q11=np.abs(usol[0,:] - u_pred[0,:]) #Absolute error when x =0 for all t
q11=torch.from_numpy(q11)
q11

#Analysis for the box plot
q12=np.abs(usol[50,:] - u_pred[50,:]) #Absolute error when x =50 for all t
q12=torch.from_numpy(q12)
q12

#Analysis for the box plot
q13=np.abs(usol[99,:] - u_pred[99,:]) #Absolute error when x =100 for all t
q13=torch.from_numpy(q13)
q13

#Analysis for the box plot
q21=np.abs(usol[:,0] - u_pred[:,0]) #Absolute error when t =0 for all x
q21=torch.from_numpy(q21)
q21

#Analysis for the box plot
q22=np.abs(usol[:,90] - u_pred[:,90]) #Absolute error when t =90 for all x
q22=torch.from_numpy(q22)
q22

#Analysis for the box plot
q23=np.abs(usol[:,179] - u_pred[:,179]) #Absolute error when t =180 for all x
q23=torch.from_numpy(q23)
q23

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

plt.plot(s,q,'*', color = 'blue', markersize = 2, label = 'Predicted Temperature (initial temperature =10degC)') #predicted
plt.plot(t,w,'o', color = 'green', markersize = 2, label = 'Exact Solution (initial temperature =10degC)') #solution


plt.xlabel(r'Time (hours)')
plt.ylabel(r'Temperature (degC)')
plt.title('Adiabatic temperature rise during cement hydration')#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.legend(loc='lower right')
#plt.axis('scaled')
plt.grid()
plt.show()

#calculate avg of absolute error values over the domain
e=[]
k1=np.abs(usol - u_pred)

kk1=k1.transpose()

for i in range(180):
    v=cal_average(torch.from_numpy(kk1)[i])
    e.append(v)

plt.plot(t,e,'-', markersize = 2) #absolute error

plt.xlabel(r'Time (hours)')
plt.ylabel(r'Error values (degC)')
plt.title('Absolute error')
plt.grid()
plt.show()

#Absolute error data for a time range:
q31=np.abs(usol[:,25:35] - u_pred[:,25:35]) #Absolute error when t =25-35 for all x
q31=torch.from_numpy(q31)
q31min=np.abs(usol[:,25] - u_pred[:,25]) #Absolute error when t =25 for all x
q31min=torch.from_numpy(q31min)
q31max=np.abs(usol[:,35] - u_pred[:,35]) #Absolute error when t =35 for all x
q31max=torch.from_numpy(q31max)

plt.plot(q31min,'--',color = 'red',label = 't=25 hours')
plt.plot(q31max,'--',color = 'red',label = 't=35 hours')


#plt.plot(x,q31max,
         #x,q31min)
#plt.fill_between(x,q31max, q31min, color='green',
                 #alpha=0.5)

plt.xlabel(r'x (μm)', fontsize=18)
plt.ylabel(r'Absolute error (℃)',fontsize=18)
plt.text(10, 1.82, 't=35 hours')
plt.text(70, 2.1, 't=25 hours')
plt.show()

#Save data
lp=np.array(q) #predicted data
lp

from numpy import savetxt
savetxt('data.csv', lp, delimiter=',')

# load numpy array from csv file
from numpy import loadtxt
# load array
data = loadtxt('data.csv', delimiter=',')
# print the array
#print(data)
from google.colab import files
files.download("data.csv")

"""Box plot: Combining absolute error results from different cases above."""

#At x=0, all t

#when collocation points/data =1%, x=0, all t:
data1_x0=[37.0485, 36.0485, 35.0486, 34.0486, 33.5486, 33.0486, 32.5487, 31.8487,
        31.0487, 30.0488, 29.0488, 28.0489, 27.0489, 26.0489, 24.5489, 24.0490,
        23.0490, 22.0490, 21.8490, 18.5491, 18.0491, 16.5492, 15.0492, 13.5492,
        12.0492, 10.5493,  9.0493,  7.5493,  6.0493,  4.5494,  3.0494,  1.5495,
         0.0495,  0.9505,  1.9505,  2.9504,  3.9504,  4.9503,  5.9503,  6.9503,
         7.9503,  8.9502, 10.4502, 11.9502, 12.9502, 13.9501, 14.9501, 15.2000,
        15.4500, 15.7000, 15.9500, 16.2000, 16.4499, 16.6999, 16.9499, 17.2999,
        17.6998, 18.0498, 18.3497, 18.7997, 19.2497, 19.4497, 19.6497, 19.8496,
        19.8996, 19.9496, 20.0496, 20.1496, 20.2495, 20.3495, 20.3894, 20.4494,
        20.5494, 20.5994, 20.6994, 20.7493, 20.8493, 20.8992, 20.9492, 21.0492,
        21.0992, 21.1492, 21.2491, 21.3491, 21.3991, 21.4491, 21.4991, 21.5490,
        21.5990, 21.6490, 21.6989, 21.7489, 21.7989, 21.8489, 21.8989, 21.9388,
        21.9398, 21.9403, 21.9408, 21.9413, 21.9418, 21.9422, 21.9423, 21.9427,
        21.9436, 21.9441, 21.9446, 21.9456, 21.9457, 21.9457, 21.9460, 21.9465,
        21.9705, 22.0079, 22.0433, 22.0768, 22.1085, 22.1382, 22.1660, 22.1953,
        22.2252, 22.2539, 22.2816, 22.3081, 22.3334, 22.3575, 22.3806, 22.4024,
        22.4231, 22.4427, 22.4611, 22.4783, 22.4947, 22.5136, 22.5318, 22.5494,
        22.5664, 22.5828, 22.5985, 22.6136, 22.6282, 22.6420, 22.6553, 22.6679,
        22.6799, 22.6913, 22.7021, 22.7123, 22.7217, 22.7307, 22.7391, 22.7491,
        22.7587, 22.7679, 22.7767, 22.7852, 22.7932, 22.8009, 22.8083, 22.8152,
        22.8217, 22.8279, 22.8337, 22.8391, 22.8441, 22.8487, 22.8530, 22.8569,
        22.9475, 22.9725, 22.9975, 23.0225, 23.0474, 23.0724, 23.0974, 23.1224,
        23.1474, 23.1724, 23.1974, 23.1974]

#when collocation points/data =5%, x=0, all t:
data5_x0=[3.5556e-01, 7.1973e-01, 1.0373e+00, 1.3072e+00, 1.0287e+00, 7.0209e-01,
        3.2794e-01, 1.0796e-01, 5.5328e-02, 5.8411e-02, 9.6893e-02, 1.6534e-01,
        2.5757e-01, 3.6679e-01, 1.4252e-02, 6.0712e-01, 7.2388e-01, 8.2974e-01,
        1.7195e+00, 5.1041e-01, 3.8788e-02, 4.3146e-01, 9.1650e-01, 1.4080e+00,
        1.8942e+00, 2.3597e+00, 2.7869e+00, 3.1568e+00, 3.4523e+00, 3.6598e+00,
        3.7728e+00, 3.7935e+00, 3.7339e+00, 3.1147e+00, 2.4636e+00, 1.8110e+00,
        1.1873e+00, 6.1914e-01, 1.2799e-01, 2.7100e-01, 5.6891e-01, 7.6215e-01,
        3.5126e-01, 1.6003e-01, 2.6593e-01, 4.5947e-01, 7.3334e-01, 3.2990e-01,
        8.0872e-03, 2.8741e-01, 5.1447e-01, 6.9489e-01, 8.3389e-01, 9.3610e-01,
        1.0056e+00, 9.4587e-01, 8.1030e-01, 7.0164e-01, 6.2243e-01, 3.7485e-01,
        1.1086e-01, 8.2123e-02, 4.0204e-02, 1.3525e-02, 7.2034e-02, 1.4801e-01,
        1.6526e-01, 1.7469e-01, 1.7705e-01, 1.7297e-01, 2.2299e-01, 2.4774e-01,
        2.2758e-01, 2.5302e-01, 2.2437e-01, 2.4205e-01, 2.0626e-01, 2.1742e-01,
        2.2562e-01, 1.8119e-01, 1.8429e-01, 1.8513e-01, 1.3384e-01, 8.0591e-02,
        7.5574e-02, 6.8817e-02, 6.0474e-02, 5.0696e-02, 3.9484e-02, 2.6990e-02,
        1.3245e-02, 1.5991e-03, 1.7542e-02, 3.4521e-02, 5.2448e-02, 6.1289e-02,
        3.2016e-02, 3.0356e-03, 2.5151e-02, 5.2605e-02, 7.9327e-02, 1.0541e-01,
        1.3122e-01, 1.5615e-01, 1.7987e-01, 2.0351e-01, 2.2660e-01, 2.4873e-01,
        2.7121e-01, 2.9321e-01, 3.1457e-01, 3.3525e-01, 3.3198e-01, 3.1499e-01,
        2.9954e-01, 2.8555e-01, 2.7312e-01, 2.6222e-01, 2.5289e-01, 2.4182e-01,
        2.2972e-01, 2.1848e-01, 2.0808e-01, 1.9856e-01, 1.8988e-01, 1.8211e-01,
        1.7519e-01, 1.6918e-01, 1.6408e-01, 1.5991e-01, 1.5662e-01, 1.5427e-01,
        1.5258e-01, 1.4821e-01, 1.4428e-01, 1.4075e-01, 1.3763e-01, 1.3498e-01,
        1.3274e-01, 1.3097e-01, 1.2963e-01, 1.2876e-01, 1.2837e-01, 1.2844e-01,
        1.2898e-01, 1.2996e-01, 1.3147e-01, 1.3345e-01, 1.3592e-01, 1.3890e-01,
        1.4218e-01, 1.4388e-01, 1.4581e-01, 1.4802e-01, 1.5050e-01, 1.5327e-01,
        1.5629e-01, 1.5963e-01, 1.6327e-01, 1.6716e-01, 1.7137e-01, 1.7587e-01,
        1.8066e-01, 1.8577e-01, 1.9120e-01, 1.9695e-01, 2.0299e-01, 2.0935e-01,
        1.2891e-01, 1.1404e-01, 9.9109e-02, 8.4119e-02, 6.9098e-02, 5.4016e-02,
        3.8873e-02, 2.3700e-02, 8.4656e-03, 6.7993e-03, 2.2095e-02, 1.2451e-02]

#when collocation points/data =10%, x=0, all t:
data10_x0=[42.0720, 41.0720, 40.0720, 39.0720, 38.5720, 38.0720, 37.5720, 36.8720,
        36.0720, 35.0720, 34.0720, 33.0720, 32.0720, 31.0720, 29.5720, 29.0720,
        28.0720, 27.0720, 26.8720, 23.5720, 23.0720, 21.5720, 20.0720, 18.5720,
        17.0720, 15.5720, 14.0720, 12.5720, 11.0720,  9.5720,  8.0720,  6.5720,
         5.0720,  4.0721,  3.0721,  2.0721,  1.0721,  0.0721,  0.9279,  1.9279,
         2.9279,  3.9279,  5.4279,  6.9279,  7.9279,  8.9279,  9.9279, 10.1779,
        10.4279, 10.6779, 10.9279, 11.1779, 11.4279, 11.6779, 11.9279, 12.2779,
        12.6779, 13.0279, 13.3279, 13.7779, 14.2279, 14.4279, 14.6279, 14.8279,
        14.8779, 14.9279, 15.0279, 15.1279, 15.2279, 15.3279, 15.3679, 15.4279,
        15.5279, 15.5778, 15.6778, 15.7278, 15.8278, 15.8778, 15.9278, 16.0278,
        16.0778, 16.1278, 16.2278, 16.3278, 16.3778, 16.4278, 16.4778, 16.5278,
        16.5778, 16.6278, 16.6778, 16.7278, 16.7778, 16.8278, 16.8778, 16.9178,
        16.9187, 16.9192, 16.9197, 16.9202, 16.9207, 16.9212, 16.9213, 16.9217,
        16.9227, 16.9232, 16.9237, 16.9247, 16.9248, 16.9249, 16.9252, 16.9257,
        16.9498, 16.9871, 17.0226, 17.0561, 17.0878, 17.1175, 17.1454, 17.1746,
        17.2045, 17.2333, 17.2610, 17.2875, 17.3128, 17.3370, 17.3600, 17.3819,
        17.4026, 17.4222, 17.4406, 17.4579, 17.4743, 17.4931, 17.5114, 17.5290,
        17.5460, 17.5624, 17.5782, 17.5933, 17.6078, 17.6217, 17.6350, 17.6476,
        17.6597, 17.6711, 17.6819, 17.6920, 17.7016, 17.7105, 17.7190, 17.7290,
        17.7386, 17.7478, 17.7566, 17.7651, 17.7732, 17.7809, 17.7882, 17.7951,
        17.8017, 17.8079, 17.8137, 17.8191, 17.8241, 17.8287, 17.8330, 17.8369,
        17.9276, 17.9526, 17.9776, 18.0026, 18.0276, 18.0526, 18.0776, 18.1026,
        18.1276, 18.1526, 18.1776, 18.1776]

#when collocation points/data =20%, x=0, all t:
data20_x0=[37.7399, 36.7399, 35.7399, 34.7399, 34.2399, 33.7399, 33.2399, 32.5399,
        31.7399, 30.7399, 29.7399, 28.7399, 27.7399, 26.7399, 25.2399, 24.7399,
        23.7399, 22.7399, 22.5399, 19.2399, 18.7399, 17.2399, 15.7399, 14.2399,
        12.7399, 11.2399,  9.7399,  8.2399,  6.7399,  5.2399,  3.7399,  2.2399,
         0.7399,  0.2601,  1.2601,  2.2601,  3.2601,  4.2601,  5.2601,  6.2601,
         7.2601,  8.2601,  9.7601, 11.2601, 12.2601, 13.2601, 14.2601, 14.5101,
        14.7601, 15.0101, 15.2601, 15.5101, 15.7601, 16.0101, 16.2601, 16.6101,
        17.0101, 17.3601, 17.6601, 18.1101, 18.5601, 18.7601, 18.9601, 19.1601,
        19.2101, 19.2601, 19.3601, 19.4601, 19.5601, 19.6601, 19.7001, 19.7601,
        19.8601, 19.9101, 20.0101, 20.0601, 20.1601, 20.2101, 20.2601, 20.3601,
        20.4101, 20.4601, 20.5601, 20.6601, 20.7101, 20.7601, 20.8101, 20.8601,
        20.9101, 20.9601, 21.0101, 21.0601, 21.1101, 21.1601, 21.2101, 21.2501,
        21.2511, 21.2516, 21.2521, 21.2526, 21.2531, 21.2536, 21.2537, 21.2541,
        21.2551, 21.2556, 21.2561, 21.2571, 21.2572, 21.2573, 21.2576, 21.2581,
        21.2822, 21.3195, 21.3550, 21.3885, 21.4202, 21.4499, 21.4778, 21.5070,
        21.5370, 21.5658, 21.5934, 21.6199, 21.6452, 21.6694, 21.6925, 21.7143,
        21.7351, 21.7546, 21.7731, 21.7903, 21.8067, 21.8256, 21.8439, 21.8615,
        21.8785, 21.8949, 21.9106, 21.9257, 21.9403, 21.9542, 21.9674, 21.9801,
        21.9921, 22.0035, 22.0143, 22.0245, 22.0340, 22.0430, 22.0514, 22.0614,
        22.0710, 22.0802, 22.0891, 22.0976, 22.1056, 22.1133, 22.1207, 22.1276,
        22.1342, 22.1404, 22.1462, 22.1516, 22.1566, 22.1613, 22.1655, 22.1694,
        22.2601, 22.2851, 22.3101, 22.3351, 22.3601, 22.3851, 22.4101, 22.4351,
        22.4601, 22.4851, 22.5101, 22.5101]

#At x=50, all t

#when collocation points/data =1%, x=50, all t
data1_x50=[37.0506, 36.0506, 35.0506, 34.0507, 33.5507, 33.0507, 32.5508, 31.8508,
        31.0508, 30.0508, 29.0509, 28.0509, 27.0509, 26.0509, 24.5510, 24.0510,
        23.0510, 22.0511, 21.8511, 18.5511, 18.0511, 16.5512, 15.0512, 13.5512,
        12.0512, 10.5512,  9.0513,  7.5513,  6.0514,  4.5514,  3.0514,  1.5514,
         0.0514,  0.9485,  1.9485,  2.9485,  3.9485,  4.9485,  5.9484,  6.9484,
         7.9483,  8.9483, 10.4483, 11.9483, 12.9483, 13.9482, 14.9482, 15.1982,
        15.4482, 15.6982, 15.9482, 16.1981, 16.4481, 16.6980, 16.9480, 17.2980,
        17.6980, 18.0480, 18.3480, 18.7979, 19.2479, 19.4478, 19.6478, 19.8478,
        19.8978, 19.9478, 20.0478, 20.1477, 20.2477, 20.3477, 20.3877, 20.4477,
        20.5477, 20.5976, 20.6976, 20.7476, 20.8475, 20.8975, 20.9475, 21.0475,
        21.0975, 21.1475, 21.2474, 21.3474, 21.3974, 21.4474, 21.4974, 21.5474,
        21.5973, 21.6473, 21.6973, 21.7473, 21.7972, 21.8472, 21.8972, 21.9372,
        21.9382, 21.9387, 21.9391, 21.9396, 21.9401, 21.9406, 21.9407, 21.9411,
        21.9421, 21.9425, 21.9430, 21.9440, 21.9441, 21.9441, 21.9444, 21.9449,
        21.9690, 22.0063, 22.0417, 22.0753, 22.1069, 22.1366, 22.1645, 22.1937,
        22.2237, 22.2524, 22.2801, 22.3065, 22.3319, 22.3560, 22.3790, 22.4009,
        22.4216, 22.4412, 22.4596, 22.4769, 22.4933, 22.5121, 22.5303, 22.5479,
        22.5649, 22.5813, 22.5970, 22.6122, 22.6267, 22.6406, 22.6538, 22.6665,
        22.6785, 22.6899, 22.7007, 22.7108, 22.7203, 22.7293, 22.7377, 22.7477,
        22.7573, 22.7665, 22.7753, 22.7838, 22.7919, 22.7995, 22.8069, 22.8138,
        22.8204, 22.8265, 22.8323, 22.8377, 22.8427, 22.8474, 22.8517, 22.8556,
        22.9461, 22.9711, 22.9961, 23.0211, 23.0461, 23.0711, 23.0961, 23.1211,
        23.1461, 23.1711, 23.1961, 23.1960]

#when collocation points/data =5%, x=50, all t
data5_x50=[3.1525e-02, 3.8306e-01, 6.8842e-01, 9.4675e-01, 6.5771e-01, 3.2172e-01,
        6.0242e-02, 2.8618e-01, 4.5325e-01, 4.5767e-01, 4.9478e-01, 5.5902e-01,
        6.4413e-01, 7.4329e-01, 3.4946e-01, 9.5554e-01, 1.0549e+00, 1.1418e+00,
        2.0115e+00, 2.3859e-01, 2.9089e-01, 1.9781e-01, 6.9931e-01, 1.2048e+00,
        1.7020e+00, 2.1757e+00, 2.6086e+00, 2.9830e+00, 3.2831e+00, 3.4973e+00,
        3.6207e+00, 3.6570e+00, 3.6190e+00, 3.0273e+00, 2.4083e+00, 1.7910e+00,
        1.2037e+00, 6.7120e-01, 2.1338e-01, 1.5576e-01, 4.2792e-01, 5.9970e-01,
        1.7157e-01, 3.5303e-01, 4.6881e-01, 6.6931e-01, 9.4754e-01, 5.4648e-01,
        2.0923e-01, 7.0648e-02, 2.9922e-01, 4.8196e-01, 6.2384e-01, 7.2925e-01,
        8.0219e-01, 7.4622e-01, 6.1438e-01, 5.0950e-01, 4.3405e-01, 1.9010e-01,
        7.0386e-02, 9.5703e-02, 1.3439e-01, 1.8506e-01, 9.6606e-02, 1.7883e-02,
        1.9592e-03, 1.3806e-02, 1.8329e-02, 1.6260e-02, 6.8148e-02, 9.4543e-02,
        7.5903e-02, 1.0269e-01, 7.5226e-02, 9.3921e-02, 5.9045e-02, 7.0874e-02,
        7.9651e-02, 3.5681e-02, 3.9087e-02, 4.0112e-02, 1.1090e-02, 6.4368e-02,
        6.9629e-02, 7.6660e-02, 8.5400e-02, 9.5667e-02, 1.0746e-01, 1.2065e-01,
        1.3513e-01, 1.5089e-01, 1.6781e-01, 1.8580e-01, 2.0485e-01, 2.1488e-01,
        1.8686e-01, 1.5922e-01, 1.3244e-01, 1.0645e-01, 8.1257e-02, 5.6793e-02,
        3.2601e-02, 9.4419e-03, 1.2538e-02, 3.4377e-02, 5.5605e-02, 7.5784e-02,
        9.6314e-02, 1.1629e-01, 1.3562e-01, 1.5419e-01, 1.4876e-01, 1.2960e-01,
        1.1188e-01, 9.5666e-02, 8.0919e-02, 6.7674e-02, 5.5959e-02, 4.2506e-02,
        2.7941e-02, 1.4223e-02, 1.3213e-03, 1.0764e-02, 2.2003e-02, 3.2394e-02,
        4.1939e-02, 5.0607e-02, 5.8397e-02, 6.5309e-02, 7.1314e-02, 7.6472e-02,
        8.0969e-02, 8.8149e-02, 9.4982e-02, 1.0138e-01, 1.0740e-01, 1.1301e-01,
        1.1818e-01, 1.2294e-01, 1.2727e-01, 1.3115e-01, 1.3463e-01, 1.3764e-01,
        1.4019e-01, 1.4232e-01, 1.4395e-01, 1.4515e-01, 1.4585e-01, 1.4611e-01,
        1.4603e-01, 1.4760e-01, 1.4896e-01, 1.5001e-01, 1.5086e-01, 1.5142e-01,
        1.5172e-01, 1.5174e-01, 1.5152e-01, 1.5105e-01, 1.5025e-01, 1.4920e-01,
        1.4786e-01, 1.4620e-01, 1.4428e-01, 1.4207e-01, 1.3954e-01, 1.3675e-01,
        2.2076e-01, 2.3920e-01, 2.5773e-01, 2.7632e-01, 2.9501e-01, 3.1372e-01,
        3.3250e-01, 3.5136e-01, 3.7026e-01, 3.8925e-01, 4.0826e-01, 4.0231e-01]

#when collocation points/data =10%, x=50, all t
data10_x50=[42.0704, 41.0705, 40.0705, 39.0705, 38.5705, 38.0705, 37.5705, 36.8705,
        36.0705, 35.0705, 34.0706, 33.0706, 32.0706, 31.0706, 29.5706, 29.0706,
        28.0706, 27.0706, 26.8706, 23.5706, 23.0706, 21.5706, 20.0706, 18.5706,
        17.0707, 15.5707, 14.0707, 12.5707, 11.0707,  9.5707,  8.0707,  6.5707,
         5.0707,  4.0707,  3.0707,  2.0707,  1.0708,  0.0708,  0.9292,  1.9292,
         2.9292,  3.9292,  5.4292,  6.9292,  7.9292,  8.9292,  9.9291, 10.1791,
        10.4291, 10.6791, 10.9291, 11.1791, 11.4291, 11.6791, 11.9291, 12.2791,
        12.6791, 13.0291, 13.3291, 13.7790, 14.2290, 14.4290, 14.6290, 14.8290,
        14.8790, 14.9290, 15.0290, 15.1290, 15.2290, 15.3290, 15.3690, 15.4290,
        15.5290, 15.5790, 15.6790, 15.7289, 15.8289, 15.8789, 15.9289, 16.0289,
        16.0789, 16.1289, 16.2289, 16.3289, 16.3789, 16.4288, 16.4788, 16.5288,
        16.5788, 16.6288, 16.6788, 16.7288, 16.7788, 16.8288, 16.8788, 16.9188,
        16.9198, 16.9203, 16.9208, 16.9212, 16.9217, 16.9222, 16.9223, 16.9227,
        16.9237, 16.9242, 16.9247, 16.9257, 16.9258, 16.9259, 16.9261, 16.9266,
        16.9507, 16.9881, 17.0235, 17.0571, 17.0887, 17.1185, 17.1463, 17.1755,
        17.2055, 17.2343, 17.2619, 17.2884, 17.3137, 17.3379, 17.3609, 17.3828,
        17.4035, 17.4231, 17.4415, 17.4588, 17.4752, 17.4940, 17.5123, 17.5299,
        17.5469, 17.5633, 17.5790, 17.5941, 17.6087, 17.6225, 17.6358, 17.6485,
        17.6605, 17.6719, 17.6827, 17.6928, 17.7024, 17.7113, 17.7198, 17.7297,
        17.7394, 17.7486, 17.7574, 17.7659, 17.7740, 17.7817, 17.7890, 17.7959,
        17.8024, 17.8086, 17.8144, 17.8198, 17.8249, 17.8295, 17.8338, 17.8377,
        17.9283, 17.9533, 17.9783, 18.0033, 18.0283, 18.0533, 18.0783, 18.1033,
        18.1283, 18.1533, 18.1783, 18.1783]

#when collocation points/data =20%, x=50, all t
data20_x50=[37.7399, 36.7399, 35.7399, 34.7399, 34.2399, 33.7399, 33.2399, 32.5399,
        31.7399, 30.7399, 29.7399, 28.7399, 27.7399, 26.7399, 25.2399, 24.7399,
        23.7399, 22.7399, 22.5399, 19.2399, 18.7399, 17.2399, 15.7399, 14.2399,
        12.7399, 11.2399,  9.7399,  8.2399,  6.7399,  5.2399,  3.7399,  2.2399,
         0.7399,  0.2601,  1.2601,  2.2601,  3.2601,  4.2601,  5.2601,  6.2601,
         7.2601,  8.2601,  9.7601, 11.2601, 12.2601, 13.2601, 14.2601, 14.5101,
        14.7601, 15.0101, 15.2601, 15.5101, 15.7601, 16.0101, 16.2601, 16.6101,
        17.0101, 17.3601, 17.6601, 18.1101, 18.5601, 18.7601, 18.9601, 19.1601,
        19.2101, 19.2601, 19.3601, 19.4601, 19.5601, 19.6601, 19.7001, 19.7601,
        19.8601, 19.9101, 20.0101, 20.0601, 20.1601, 20.2101, 20.2601, 20.3601,
        20.4101, 20.4601, 20.5601, 20.6601, 20.7101, 20.7601, 20.8101, 20.8601,
        20.9101, 20.9601, 21.0101, 21.0601, 21.1101, 21.1601, 21.2101, 21.2501,
        21.2511, 21.2516, 21.2521, 21.2526, 21.2531, 21.2536, 21.2537, 21.2541,
        21.2551, 21.2556, 21.2561, 21.2571, 21.2572, 21.2573, 21.2576, 21.2581,
        21.2822, 21.3195, 21.3550, 21.3885, 21.4202, 21.4499, 21.4778, 21.5070,
        21.5370, 21.5658, 21.5934, 21.6199, 21.6452, 21.6694, 21.6924, 21.7143,
        21.7350, 21.7546, 21.7730, 21.7903, 21.8067, 21.8256, 21.8438, 21.8614,
        21.8785, 21.8948, 21.9106, 21.9257, 21.9403, 21.9542, 21.9674, 21.9801,
        21.9921, 22.0035, 22.0143, 22.0245, 22.0340, 22.0430, 22.0514, 22.0614,
        22.0710, 22.0802, 22.0891, 22.0976, 22.1056, 22.1133, 22.1207, 22.1276,
        22.1342, 22.1404, 22.1462, 22.1516, 22.1566, 22.1613, 22.1655, 22.1694,
        22.2601, 22.2851, 22.3101, 22.3351, 22.3601, 22.3851, 22.4101, 22.4351,
        22.4601, 22.4851, 22.5101, 22.5101]

#At x=100, all t

#when collocation points/data =1%, x=100, all t
data1_x99=[37.0527, 36.0527, 35.0528, 34.0528, 33.5528, 33.0528, 32.5529, 31.8529,
        31.0529, 30.0529, 29.0529, 28.0530, 27.0530, 26.0530, 24.5531, 24.0531,
        23.0531, 22.0531, 21.8531, 18.5532, 18.0532, 16.5533, 15.0533, 13.5533,
        12.0533, 10.5533,  9.0533,  7.5534,  6.0534,  4.5534,  3.0534,  1.5534,
         0.0535,  0.9465,  1.9465,  2.9464,  3.9464,  4.9464,  5.9464,  6.9464,
         7.9464,  8.9463, 10.4463, 11.9463, 12.9463, 13.9463, 14.9463, 15.1962,
        15.4462, 15.6962, 15.9461, 16.1961, 16.4461, 16.6961, 16.9461, 17.2961,
        17.6960, 18.0460, 18.3460, 18.7960, 19.2460, 19.4460, 19.6460, 19.8459,
        19.8959, 19.9459, 20.0458, 20.1458, 20.2458, 20.3458, 20.3858, 20.4458,
        20.5458, 20.5957, 20.6957, 20.7457, 20.8457, 20.8956, 20.9456, 21.0456,
        21.0956, 21.1456, 21.2456, 21.3456, 21.3955, 21.4455, 21.4955, 21.5455,
        21.5955, 21.6455, 21.6955, 21.7455, 21.7954, 21.8454, 21.8954, 21.9354,
        21.9363, 21.9368, 21.9373, 21.9378, 21.9383, 21.9388, 21.9389, 21.9393,
        21.9403, 21.9407, 21.9412, 21.9422, 21.9423, 21.9424, 21.9427, 21.9432,
        21.9672, 22.0046, 22.0400, 22.0735, 22.1052, 22.1349, 22.1628, 22.1920,
        22.2219, 22.2507, 22.2783, 22.3048, 22.3302, 22.3543, 22.3773, 22.3992,
        22.4199, 22.4395, 22.4579, 22.4752, 22.4916, 22.5104, 22.5287, 22.5463,
        22.5633, 22.5796, 22.5954, 22.6105, 22.6250, 22.6389, 22.6522, 22.6649,
        22.6768, 22.6882, 22.6990, 22.7092, 22.7188, 22.7277, 22.7361, 22.7461,
        22.7557, 22.7649, 22.7738, 22.7822, 22.7903, 22.7980, 22.8053, 22.8122,
        22.8188, 22.8250, 22.8308, 22.8362, 22.8412, 22.8459, 22.8501, 22.8540,
        22.9446, 22.9696, 22.9946, 23.0196, 23.0446, 23.0696, 23.0946, 23.1195,
        23.1445, 23.1695, 23.1945, 23.1945]

#when collocation points/data =5%, x=100, all t
data5_x99=[1.6495e-01, 1.8079e-01, 4.8041e-01, 7.3309e-01, 4.3863e-01, 9.7382e-02,
        2.8943e-01, 5.1979e-01, 6.9070e-01, 6.9833e-01, 7.3782e-01, 8.0359e-01,
        8.8919e-01, 9.8779e-01, 5.9229e-01, 1.1956e+00, 1.2911e+00, 1.3731e+00,
        2.2372e+00, 1.9104e-02, 5.0381e-01, 8.6670e-03, 4.9887e-01, 1.0096e+00,
        1.5108e+00, 1.9870e+00, 2.4209e+00, 2.7949e+00, 3.0935e+00, 3.3059e+00,
        3.4280e+00, 3.4645e+00, 3.4290e+00, 2.8427e+00, 2.2322e+00, 1.6263e+00,
        1.0527e+00, 5.3571e-01, 9.4330e-02, 2.5818e-01, 5.1410e-01, 6.7041e-01,
        2.2797e-01, 3.0969e-01, 4.3719e-01, 6.4807e-01, 9.3549e-01, 5.4242e-01,
        2.1216e-01, 6.1768e-02, 2.8516e-01, 4.6350e-01, 6.0156e-01, 7.0374e-01,
        7.7396e-01, 7.1564e-01, 5.8185e-01, 4.7526e-01, 3.9834e-01, 1.5317e-01,
        1.0829e-01, 1.3446e-01, 1.7388e-01, 2.2513e-01, 1.3716e-01, 5.8899e-02,
        3.9423e-02, 2.7911e-02, 2.3633e-02, 2.5977e-02, 2.5729e-02, 5.1910e-02,
        3.3118e-02, 5.9717e-02, 3.2074e-02, 5.0586e-02, 1.5527e-02, 2.7203e-02,
        3.5828e-02, 8.3557e-03, 5.1331e-03, 4.3213e-03, 5.5768e-02, 1.0929e-01,
        1.1479e-01, 1.2213e-01, 1.3115e-01, 1.4175e-01, 1.5391e-01, 1.6741e-01,
        1.8225e-01, 1.9838e-01, 2.1566e-01, 2.3411e-01, 2.5359e-01, 2.6408e-01,
        2.3651e-01, 2.0936e-01, 1.8310e-01, 1.5763e-01, 1.3295e-01, 1.0904e-01,
        8.5458e-02, 6.2848e-02, 4.1509e-02, 2.0311e-02, 3.0737e-04, 1.9815e-02,
        3.9674e-02, 5.8983e-02, 7.7573e-02, 9.5445e-02, 8.9279e-02, 6.9358e-02,
        5.0908e-02, 3.3898e-02, 1.8358e-02, 4.3192e-03, 8.2191e-03, 2.2527e-02,
        3.7916e-02, 5.2488e-02, 6.6306e-02, 7.9276e-02, 9.1430e-02, 1.0274e-01,
        1.1323e-01, 1.2287e-01, 1.3164e-01, 1.3950e-01, 1.4654e-01, 1.5267e-01,
        1.5821e-01, 1.6646e-01, 1.7430e-01, 1.8179e-01, 1.8888e-01, 1.9559e-01,
        2.0186e-01, 2.0778e-01, 2.1323e-01, 2.1828e-01, 2.2292e-01, 2.2712e-01,
        2.3088e-01, 2.3421e-01, 2.3709e-01, 2.3951e-01, 2.4149e-01, 2.4300e-01,
        2.4424e-01, 2.4711e-01, 2.4979e-01, 2.5218e-01, 2.5438e-01, 2.5631e-01,
        2.5799e-01, 2.5943e-01, 2.6062e-01, 2.6155e-01, 2.6222e-01, 2.6264e-01,
        2.6276e-01, 2.6262e-01, 2.6223e-01, 2.6155e-01, 2.6057e-01, 2.5934e-01,
        3.4491e-01, 3.6497e-01, 3.8508e-01, 4.0532e-01, 4.2562e-01, 4.4601e-01,
        4.6647e-01, 4.8701e-01, 5.0765e-01, 5.2834e-01, 5.4913e-01, 5.4495e-01]

#when collocation points/data =10%, x=100, all t
data10_x99=[42.0691, 41.0691, 40.0691, 39.0692, 38.5692, 38.0692, 37.5692, 36.8692,
        36.0692, 35.0692, 34.0692, 33.0693, 32.0693, 31.0693, 29.5693, 29.0693,
        28.0693, 27.0693, 26.8694, 23.5694, 23.0694, 21.5694, 20.0694, 18.5694,
        17.0695, 15.5695, 14.0695, 12.5695, 11.0695,  9.5695,  8.0695,  6.5695,
         5.0695,  4.0696,  3.0696,  2.0696,  1.0696,  0.0696,  0.9304,  1.9304,
         2.9304,  3.9303,  5.4303,  6.9303,  7.9303,  8.9303,  9.9303, 10.1802,
        10.4302, 10.6802, 10.9302, 11.1802, 11.4302, 11.6802, 11.9302, 12.2802,
        12.6802, 13.0301, 13.3301, 13.7801, 14.2301, 14.4301, 14.6301, 14.8301,
        14.8801, 14.9301, 15.0301, 15.1300, 15.2300, 15.3300, 15.3700, 15.4300,
        15.5300, 15.5799, 15.6799, 15.7299, 15.8299, 15.8799, 15.9299, 16.0299,
        16.0799, 16.1299, 16.2299, 16.3298, 16.3798, 16.4298, 16.4798, 16.5298,
        16.5798, 16.6297, 16.6797, 16.7297, 16.7797, 16.8297, 16.8797, 16.9197,
        16.9207, 16.9212, 16.9217, 16.9222, 16.9227, 16.9231, 16.9232, 16.9236,
        16.9246, 16.9251, 16.9256, 16.9266, 16.9267, 16.9268, 16.9270, 16.9275,
        16.9516, 16.9889, 17.0244, 17.0579, 17.0896, 17.1193, 17.1471, 17.1764,
        17.2063, 17.2351, 17.2627, 17.2892, 17.3145, 17.3387, 17.3617, 17.3836,
        17.4043, 17.4239, 17.4423, 17.4596, 17.4760, 17.4948, 17.5131, 17.5307,
        17.5477, 17.5641, 17.5798, 17.5949, 17.6094, 17.6233, 17.6366, 17.6492,
        17.6612, 17.6727, 17.6834, 17.6936, 17.7032, 17.7121, 17.7205, 17.7305,
        17.7401, 17.7493, 17.7582, 17.7666, 17.7747, 17.7824, 17.7897, 17.7966,
        17.8032, 17.8094, 17.8152, 17.8205, 17.8256, 17.8302, 17.8345, 17.8384,
        17.9290, 17.9540, 17.9790, 18.0040, 18.0290, 18.0540, 18.0790, 18.1040,
        18.1290, 18.1540, 18.1790, 18.1789]

#when collocation points/data =20%, x=100, all t
data20_x99=[37.7399, 36.7399, 35.7399, 34.7399, 34.2399, 33.7399, 33.2399, 32.5399,
        31.7399, 30.7399, 29.7399, 28.7399, 27.7399, 26.7399, 25.2399, 24.7399,
        23.7399, 22.7399, 22.5399, 19.2399, 18.7399, 17.2399, 15.7399, 14.2399,
        12.7399, 11.2399,  9.7399,  8.2399,  6.7399,  5.2399,  3.7399,  2.2399,
         0.7399,  0.2601,  1.2601,  2.2601,  3.2601,  4.2601,  5.2601,  6.2601,
         7.2601,  8.2601,  9.7601, 11.2601, 12.2601, 13.2601, 14.2601, 14.5101,
        14.7601, 15.0101, 15.2601, 15.5101, 15.7601, 16.0101, 16.2601, 16.6101,
        17.0101, 17.3601, 17.6601, 18.1101, 18.5601, 18.7601, 18.9601, 19.1601,
        19.2101, 19.2601, 19.3601, 19.4601, 19.5601, 19.6601, 19.7001, 19.7601,
        19.8601, 19.9101, 20.0101, 20.0601, 20.1601, 20.2101, 20.2601, 20.3601,
        20.4101, 20.4601, 20.5601, 20.6601, 20.7101, 20.7601, 20.8101, 20.8601,
        20.9101, 20.9601, 21.0101, 21.0601, 21.1101, 21.1601, 21.2101, 21.2501,
        21.2511, 21.2516, 21.2521, 21.2526, 21.2531, 21.2536, 21.2537, 21.2541,
        21.2551, 21.2556, 21.2561, 21.2571, 21.2572, 21.2573, 21.2576, 21.2581,
        21.2822, 21.3195, 21.3550, 21.3885, 21.4202, 21.4499, 21.4778, 21.5070,
        21.5370, 21.5658, 21.5934, 21.6199, 21.6452, 21.6694, 21.6925, 21.7143,
        21.7351, 21.7546, 21.7731, 21.7903, 21.8067, 21.8256, 21.8439, 21.8615,
        21.8785, 21.8949, 21.9106, 21.9258, 21.9403, 21.9542, 21.9675, 21.9801,
        21.9921, 22.0036, 22.0143, 22.0245, 22.0341, 22.0430, 22.0515, 22.0614,
        22.0711, 22.0803, 22.0891, 22.0976, 22.1057, 22.1134, 22.1207, 22.1276,
        22.1342, 22.1404, 22.1462, 22.1516, 22.1566, 22.1613, 22.1656, 22.1695,
        22.2601, 22.2851, 22.3101, 22.3351, 22.3601, 22.3851, 22.4101, 22.4351,
        22.4601, 22.4851, 22.5101, 22.5101]

#At t=0, all x

#when collocation points/data =1%, t=0, all x
data1_t0=[37.0485, 37.0485, 37.0486, 37.0486, 37.0486, 37.0487, 37.0487, 37.0487,
        37.0488, 37.0489, 37.0489, 37.0489, 37.0490, 37.0490, 37.0490, 37.0491,
        37.0492, 37.0492, 37.0492, 37.0493, 37.0493, 37.0493, 37.0494, 37.0494,
        37.0495, 37.0495, 37.0496, 37.0496, 37.0497, 37.0497, 37.0497, 37.0498,
        37.0498, 37.0498, 37.0499, 37.0500, 37.0500, 37.0500, 37.0501, 37.0501,
        37.0501, 37.0502, 37.0503, 37.0503, 37.0503, 37.0504, 37.0504, 37.0504,
        37.0505, 37.0505, 37.0506, 37.0506, 37.0507, 37.0507, 37.0508, 37.0508,
        37.0508, 37.0509, 37.0509, 37.0509, 37.0510, 37.0511, 37.0511, 37.0511,
        37.0512, 37.0512, 37.0512, 37.0513, 37.0514, 37.0514, 37.0514, 37.0515,
        37.0515, 37.0515, 37.0516, 37.0517, 37.0517, 37.0517, 37.0518, 37.0518,
        37.0518, 37.0519, 37.0520, 37.0520, 37.0520, 37.0521, 37.0522, 37.0522,
        37.0522, 37.0523, 37.0523, 37.0523, 37.0524, 37.0525, 37.0525, 37.0525,
        37.0526, 37.0526, 37.0526, 37.0527]

#when collocation points/data =5%, t=0, all x
data5_t0=[3.5556e-01, 3.5135e-01, 3.4689e-01, 3.4229e-01, 3.3743e-01, 3.3243e-01,
        3.2724e-01, 3.2187e-01, 3.1631e-01, 3.1067e-01, 3.0484e-01, 2.9889e-01,
        2.9282e-01, 2.8662e-01, 2.8033e-01, 2.7393e-01, 2.6743e-01, 2.6083e-01,
        2.5418e-01, 2.4741e-01, 2.4060e-01, 2.3373e-01, 2.2681e-01, 2.1982e-01,
        2.1280e-01, 2.0575e-01, 1.9867e-01, 1.9156e-01, 1.8442e-01, 1.7728e-01,
        1.7010e-01, 1.6293e-01, 1.5579e-01, 1.4862e-01, 1.4145e-01, 1.3434e-01,
        1.2720e-01, 1.2012e-01, 1.1304e-01, 1.0599e-01, 9.8969e-02, 9.2010e-02,
        8.5052e-02, 7.8186e-02, 7.1350e-02, 6.4575e-02, 5.7831e-02, 5.1147e-02,
        4.4556e-02, 3.7994e-02, 3.1525e-02, 2.5116e-02, 1.8799e-02, 1.2512e-02,
        6.3171e-03, 2.1362e-04, 5.7983e-03, 1.1719e-02, 1.7578e-02, 2.3315e-02,
        2.8992e-02, 3.4576e-02, 4.0039e-02, 4.5441e-02, 5.0720e-02, 5.5908e-02,
        6.1005e-02, 6.5979e-02, 7.0831e-02, 7.5623e-02, 8.0261e-02, 8.4839e-02,
        8.9294e-02, 9.3628e-02, 9.7839e-02, 1.0196e-01, 1.0596e-01, 1.0983e-01,
        1.1359e-01, 1.1722e-01, 1.2076e-01, 1.2418e-01, 1.2744e-01, 1.3065e-01,
        1.3370e-01, 1.3663e-01, 1.3943e-01, 1.4215e-01, 1.4471e-01, 1.4716e-01,
        1.4951e-01, 1.5170e-01, 1.5378e-01, 1.5576e-01, 1.5759e-01, 1.5930e-01,
        1.6089e-01, 1.6238e-01, 1.6373e-01, 1.6495e-01]

#when collocation points/data =10%, t=0, all x
data10_t0=[42.0720, 42.0719, 42.0719, 42.0718, 42.0718, 42.0718, 42.0717, 42.0717,
        42.0717, 42.0717, 42.0716, 42.0716, 42.0716, 42.0715, 42.0715, 42.0715,
        42.0714, 42.0714, 42.0714, 42.0714, 42.0714, 42.0713, 42.0713, 42.0712,
        42.0712, 42.0712, 42.0712, 42.0711, 42.0711, 42.0710, 42.0710, 42.0710,
        42.0710, 42.0709, 42.0709, 42.0709, 42.0709, 42.0708, 42.0708, 42.0708,
        42.0707, 42.0707, 42.0707, 42.0706, 42.0706, 42.0706, 42.0706, 42.0705,
        42.0705, 42.0705, 42.0704, 42.0704, 42.0704, 42.0704, 42.0703, 42.0703,
        42.0703, 42.0703, 42.0703, 42.0702, 42.0702, 42.0702, 42.0701, 42.0701,
        42.0701, 42.0700, 42.0700, 42.0700, 42.0699, 42.0699, 42.0699, 42.0699,
        42.0698, 42.0698, 42.0698, 42.0698, 42.0697, 42.0697, 42.0697, 42.0696,
        42.0696, 42.0696, 42.0696, 42.0695, 42.0695, 42.0695, 42.0695, 42.0695,
        42.0694, 42.0694, 42.0693, 42.0693, 42.0693, 42.0693, 42.0693, 42.0692,
        42.0692, 42.0692, 42.0691, 42.0691]

#when collocation points/data =20%, t=0, all x
data20_t0=[37.7399, 37.7399, 37.7399, 37.7399, 37.7399, 37.7399, 37.7399, 37.7399,
        37.7399, 37.7399, 37.7399, 37.7399, 37.7399, 37.7399, 37.7399, 37.7399,
        37.7399, 37.7399, 37.7399, 37.7399, 37.7399, 37.7399, 37.7399, 37.7399,
        37.7399, 37.7399, 37.7399, 37.7399, 37.7399, 37.7399, 37.7399, 37.7399,
        37.7399, 37.7399, 37.7399, 37.7399, 37.7399, 37.7399, 37.7399, 37.7399,
        37.7399, 37.7399, 37.7399, 37.7399, 37.7399, 37.7399, 37.7399, 37.7399,
        37.7399, 37.7399, 37.7399, 37.7399, 37.7399, 37.7399, 37.7399, 37.7399,
        37.7399, 37.7399, 37.7399, 37.7399, 37.7399, 37.7399, 37.7399, 37.7399,
        37.7399, 37.7399, 37.7399, 37.7399, 37.7399, 37.7399, 37.7399, 37.7399,
        37.7399, 37.7399, 37.7399, 37.7399, 37.7399, 37.7399, 37.7399, 37.7399,
        37.7399, 37.7399, 37.7399, 37.7399, 37.7399, 37.7399, 37.7399, 37.7399,
        37.7399, 37.7399, 37.7399, 37.7399, 37.7399, 37.7399, 37.7399, 37.7399,
        37.7399, 37.7399, 37.7399, 37.7399]

#At t=90, all x

#when collocation points/data =1%, t=90, all x
data1_t90=[21.6989, 21.6989, 21.6989, 21.6989, 21.6988, 21.6988, 21.6988, 21.6988,
        21.6987, 21.6987, 21.6986, 21.6986, 21.6986, 21.6985, 21.6985, 21.6985,
        21.6985, 21.6984, 21.6984, 21.6983, 21.6983, 21.6983, 21.6983, 21.6982,
        21.6982, 21.6982, 21.6981, 21.6981, 21.6980, 21.6980, 21.6980, 21.6980,
        21.6979, 21.6978, 21.6978, 21.6978, 21.6978, 21.6977, 21.6977, 21.6977,
        21.6977, 21.6976, 21.6975, 21.6975, 21.6975, 21.6975, 21.6974, 21.6974,
        21.6974, 21.6973, 21.6973, 21.6972, 21.6972, 21.6972, 21.6972, 21.6971,
        21.6971, 21.6971, 21.6970, 21.6970, 21.6969, 21.6969, 21.6969, 21.6968,
        21.6968, 21.6967, 21.6967, 21.6967, 21.6966, 21.6966, 21.6966, 21.6965,
        21.6965, 21.6964, 21.6964, 21.6964, 21.6963, 21.6963, 21.6963, 21.6963,
        21.6962, 21.6961, 21.6961, 21.6961, 21.6960, 21.6960, 21.6960, 21.6960,
        21.6959, 21.6958, 21.6958, 21.6958, 21.6957, 21.6957, 21.6956, 21.6956,
        21.6956, 21.6955, 21.6955, 21.6955]

#when collocation points/data =5%, t=90, all x
data5_t90=[0.0132, 0.0082, 0.0033, 0.0015, 0.0062, 0.0109, 0.0154, 0.0197, 0.0240,
        0.0283, 0.0324, 0.0364, 0.0403, 0.0441, 0.0479, 0.0515, 0.0551, 0.0586,
        0.0620, 0.0652, 0.0685, 0.0716, 0.0747, 0.0777, 0.0806, 0.0834, 0.0862,
        0.0889, 0.0916, 0.0941, 0.0966, 0.0991, 0.1014, 0.1038, 0.1060, 0.1082,
        0.1103, 0.1124, 0.1144, 0.1164, 0.1183, 0.1202, 0.1221, 0.1238, 0.1256,
        0.1273, 0.1289, 0.1306, 0.1321, 0.1337, 0.1351, 0.1366, 0.1380, 0.1394,
        0.1408, 0.1421, 0.1434, 0.1447, 0.1459, 0.1471, 0.1483, 0.1494, 0.1505,
        0.1516, 0.1527, 0.1538, 0.1548, 0.1558, 0.1568, 0.1578, 0.1588, 0.1597,
        0.1606, 0.1616, 0.1625, 0.1634, 0.1642, 0.1651, 0.1659, 0.1667, 0.1676,
        0.1684, 0.1693, 0.1700, 0.1708, 0.1716, 0.1724, 0.1732, 0.1740, 0.1747,
        0.1755, 0.1762, 0.1770, 0.1778, 0.1785, 0.1793, 0.1800, 0.1808, 0.1815,
        0.1823]

#when collocation points/data =10%, t=90, all x
data10_t90=[16.6778, 16.6778, 16.6778, 16.6779, 16.6779, 16.6779, 16.6779, 16.6779,
        16.6779, 16.6780, 16.6780, 16.6780, 16.6780, 16.6780, 16.6781, 16.6781,
        16.6781, 16.6781, 16.6782, 16.6782, 16.6782, 16.6782, 16.6783, 16.6783,
        16.6783, 16.6783, 16.6783, 16.6783, 16.6784, 16.6784, 16.6784, 16.6784,
        16.6785, 16.6785, 16.6785, 16.6785, 16.6785, 16.6786, 16.6786, 16.6786,
        16.6786, 16.6786, 16.6786, 16.6786, 16.6787, 16.6787, 16.6787, 16.6788,
        16.6788, 16.6788, 16.6788, 16.6788, 16.6789, 16.6789, 16.6789, 16.6789,
        16.6790, 16.6790, 16.6790, 16.6790, 16.6790, 16.6790, 16.6790, 16.6791,
        16.6791, 16.6791, 16.6791, 16.6791, 16.6792, 16.6792, 16.6792, 16.6792,
        16.6792, 16.6793, 16.6793, 16.6793, 16.6793, 16.6793, 16.6794, 16.6794,
        16.6794, 16.6794, 16.6794, 16.6794, 16.6794, 16.6795, 16.6795, 16.6795,
        16.6795, 16.6796, 16.6796, 16.6796, 16.6796, 16.6796, 16.6797, 16.6797,
        16.6797, 16.6797, 16.6797, 16.6797]

#when collocation points/data =20%, t=90, all x
data20_t90=[21.0101, 21.0101, 21.0101, 21.0101, 21.0101, 21.0101, 21.0101, 21.0101,
        21.0101, 21.0101, 21.0101, 21.0101, 21.0101, 21.0101, 21.0101, 21.0101,
        21.0101, 21.0101, 21.0101, 21.0101, 21.0101, 21.0101, 21.0101, 21.0101,
        21.0101, 21.0101, 21.0101, 21.0101, 21.0101, 21.0101, 21.0101, 21.0101,
        21.0101, 21.0101, 21.0101, 21.0101, 21.0101, 21.0101, 21.0101, 21.0101,
        21.0101, 21.0101, 21.0101, 21.0101, 21.0101, 21.0101, 21.0101, 21.0101,
        21.0101, 21.0101, 21.0101, 21.0101, 21.0101, 21.0101, 21.0101, 21.0101,
        21.0101, 21.0101, 21.0101, 21.0101, 21.0101, 21.0101, 21.0101, 21.0101,
        21.0101, 21.0101, 21.0101, 21.0101, 21.0101, 21.0101, 21.0101, 21.0101,
        21.0101, 21.0101, 21.0101, 21.0101, 21.0101, 21.0101, 21.0101, 21.0101,
        21.0101, 21.0101, 21.0101, 21.0101, 21.0101, 21.0101, 21.0101, 21.0101,
        21.0101, 21.0101, 21.0101, 21.0101, 21.0101, 21.0101, 21.0101, 21.0101,
        21.0101, 21.0101, 21.0101, 21.0101]

#At t=180, all x

#when collocation points/data =1%, t=180, all x
data1_t179=[23.1974, 23.1973, 23.1973, 23.1973, 23.1972, 23.1972, 23.1972, 23.1972,
        23.1972, 23.1971, 23.1971, 23.1971, 23.1971, 23.1971, 23.1970, 23.1970,
        23.1969, 23.1969, 23.1969, 23.1969, 23.1969, 23.1968, 23.1968, 23.1967,
        23.1967, 23.1967, 23.1967, 23.1967, 23.1966, 23.1966, 23.1966, 23.1966,
        23.1966, 23.1965, 23.1965, 23.1964, 23.1964, 23.1964, 23.1964, 23.1963,
        23.1963, 23.1963, 23.1963, 23.1963, 23.1962, 23.1962, 23.1961, 23.1961,
        23.1961, 23.1961, 23.1960, 23.1960, 23.1960, 23.1960, 23.1960, 23.1959,
        23.1959, 23.1958, 23.1958, 23.1958, 23.1958, 23.1957, 23.1957, 23.1956,
        23.1956, 23.1956, 23.1956, 23.1955, 23.1955, 23.1955, 23.1955, 23.1954,
        23.1954, 23.1953, 23.1953, 23.1953, 23.1953, 23.1953, 23.1952, 23.1952,
        23.1952, 23.1951, 23.1951, 23.1950, 23.1950, 23.1950, 23.1950, 23.1949,
        23.1949, 23.1949, 23.1949, 23.1948, 23.1948, 23.1947, 23.1947, 23.1947,
        23.1947, 23.1946, 23.1946, 23.1945]

#when collocation points/data =5%, t=180, all x
data5_t179=[0.0125, 0.0237, 0.0348, 0.0457, 0.0565, 0.0671, 0.0776, 0.0880, 0.0982,
        0.1082, 0.1181, 0.1279, 0.1375, 0.1470, 0.1562, 0.1654, 0.1744, 0.1833,
        0.1920, 0.2006, 0.2090, 0.2173, 0.2255, 0.2335, 0.2414, 0.2492, 0.2568,
        0.2643, 0.2716, 0.2788, 0.2859, 0.2928, 0.2997, 0.3063, 0.3129, 0.3194,
        0.3257, 0.3319, 0.3380, 0.3439, 0.3498, 0.3555, 0.3612, 0.3667, 0.3721,
        0.3774, 0.3826, 0.3877, 0.3926, 0.3975, 0.4023, 0.4070, 0.4116, 0.4161,
        0.4205, 0.4248, 0.4290, 0.4331, 0.4372, 0.4412, 0.4450, 0.4488, 0.4525,
        0.4561, 0.4597, 0.4632, 0.4666, 0.4699, 0.4731, 0.4763, 0.4794, 0.4825,
        0.4855, 0.4883, 0.4912, 0.4940, 0.4967, 0.4994, 0.5020, 0.5045, 0.5070,
        0.5095, 0.5118, 0.5142, 0.5164, 0.5186, 0.5208, 0.5229, 0.5250, 0.5270,
        0.5290, 0.5309, 0.5328, 0.5347, 0.5365, 0.5383, 0.5400, 0.5417, 0.5433,
        0.5450]

#when collocation points/data =10%, t=180, all x
data10_t179=[18.1776, 18.1776, 18.1776, 18.1776, 18.1776, 18.1776, 18.1776, 18.1776,
        18.1777, 18.1777, 18.1777, 18.1777, 18.1777, 18.1777, 18.1777, 18.1778,
        18.1778, 18.1778, 18.1778, 18.1778, 18.1778, 18.1779, 18.1779, 18.1779,
        18.1779, 18.1779, 18.1779, 18.1779, 18.1779, 18.1780, 18.1780, 18.1780,
        18.1780, 18.1780, 18.1780, 18.1780, 18.1781, 18.1781, 18.1781, 18.1781,
        18.1781, 18.1782, 18.1782, 18.1782, 18.1782, 18.1782, 18.1782, 18.1782,
        18.1782, 18.1783, 18.1783, 18.1783, 18.1783, 18.1783, 18.1783, 18.1783,
        18.1783, 18.1783, 18.1784, 18.1784, 18.1784, 18.1784, 18.1784, 18.1784,
        18.1785, 18.1785, 18.1785, 18.1785, 18.1785, 18.1785, 18.1785, 18.1785,
        18.1786, 18.1786, 18.1786, 18.1786, 18.1786, 18.1786, 18.1786, 18.1786,
        18.1787, 18.1787, 18.1787, 18.1787, 18.1787, 18.1787, 18.1788, 18.1788,
        18.1788, 18.1788, 18.1788, 18.1788, 18.1788, 18.1788, 18.1789, 18.1789,
        18.1789, 18.1789, 18.1789, 18.1789]

#when collocation points/data =20%, t=180, all x
data20_t179=[22.5101, 22.5101, 22.5101, 22.5101, 22.5101, 22.5101, 22.5101, 22.5101,
        22.5101, 22.5101, 22.5101, 22.5101, 22.5101, 22.5101, 22.5101, 22.5101,
        22.5101, 22.5101, 22.5101, 22.5101, 22.5101, 22.5101, 22.5101, 22.5101,
        22.5101, 22.5101, 22.5101, 22.5101, 22.5101, 22.5101, 22.5101, 22.5101,
        22.5101, 22.5101, 22.5101, 22.5101, 22.5101, 22.5101, 22.5101, 22.5101,
        22.5101, 22.5101, 22.5101, 22.5101, 22.5101, 22.5101, 22.5101, 22.5101,
        22.5101, 22.5101, 22.5101, 22.5101, 22.5101, 22.5101, 22.5101, 22.5101,
        22.5101, 22.5101, 22.5101, 22.5101, 22.5101, 22.5101, 22.5101, 22.5101,
        22.5101, 22.5101, 22.5101, 22.5101, 22.5101, 22.5101, 22.5101, 22.5101,
        22.5101, 22.5101, 22.5101, 22.5101, 22.5101, 22.5101, 22.5101, 22.5101,
        22.5101, 22.5101, 22.5101, 22.5101, 22.5101, 22.5101, 22.5101, 22.5101,
        22.5101, 22.5101, 22.5101, 22.5101, 22.5101, 22.5101, 22.5101, 22.5101,
        22.5101, 22.5101, 22.5101, 22.5101]

import matplotlib.pyplot as plt
import matplotlib.pyplot as plot

#data = [data1_x0, data5_x0, data10_x0, data20_x0] #when x=0, all t
#data = [data1_x50, data5_x50, data10_x50, data20_x50] #when x=50, all t
#data = [data1_x99, data5_x99, data10_x99, data20_x99] #when x=100, all t
#data = [data1_t0, data5_t0, data10_t0, data20_t0] #when t=0, all x
#data = [data1_t90, data5_t90, data10_t90, data20_t90] #when t=90, all x
data = [data1_t179, data5_t179, data10_t179, data20_t179] #when t=180, all x

fig = plt.figure(figsize =(6, 4))

# Creating axes instance
ax = fig.add_axes([0, 0, 1, 1])

# Creating plot
bp = ax.boxplot(data,showmeans=True, labels=["1% data","5% data","10% data","20% data"])

# Add a title and labels
plt.title(r'Absolute error $|T(x,t)- \hat T(x,t)| ℃$', fontsize=18)
plt.ylabel("Error values (℃)",fontsize=18)

# show plot
plt.show()

# Histogram and density for temperature difference between prediction and experiments:
import seaborn as sns
import pandas as pd
A=[0.44601,0.552809,0.286738,0.133766, 2.277267,0.59465,0.827232,0.192062,0.31135,0.008449,0.491551,0.691551] #Initial temp. 10degC
B=[0.742302,12.33691,26.15567,22.47528,9.958684,3.274205,1.469772,0.38018,0.332956,1.011043,0.76049,0.7483] #Initial temp. 20degC
C=[2.29927,21.49039,34.13788,22.1894,11.39802,4.55912,1.561702,0.592792,0.744782,1.000,0.4856,0.073] #Initial temp. 30degC
df = pd.DataFrame(data=A)
sns.set_theme(style='white')
g=sns.distplot(df, hist=True, kde=True,
             bins=int(180/5), color = 'red',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4});
g.set_title('Temperature difference')
plt.legend(labels=['Initial Temp. 10degC'])

#Temperature data for different initial curing temp.
time = list(range(1, 181))
# Initial temp. 10degC
T_10 = [
    9.93017, 10.43187, 11.03132, 11.73843, 12.55979, 13.49674, 14.5437, 15.68707,
    16.90517, 18.16971, 19.44847, 20.70887, 21.92172, 23.06428, 24.12212, 25.08975,
    25.97009, 26.77332, 27.51549, 28.21709, 28.90186, 29.59567, 30.32548, 31.11817,
    31.99905, 32.99026, 34.10875, 35.36442, 36.75851, 38.28274, 39.91954, 41.64324,
    43.42254, 45.22339, 47.01214, 48.75817, 50.4359, 52.02578, 53.51443, 54.89433,
    56.16283, 57.32115, 58.37342, 59.32559, 60.18482, 60.9588, 61.65527, 62.28176,
    62.84547, 63.353, 63.81047, 64.22336, 64.59659, 64.93455, 65.24118, 65.51996,
    65.77393, 66.00578, 66.21794, 66.41246, 66.5912, 66.75587, 66.90781, 67.04835,
    67.1786, 67.29962, 67.41225, 67.51729, 67.61542, 67.70734, 67.79353, 67.87452,
    67.95076, 68.02268, 68.09061, 68.15487, 68.21576, 68.27354, 68.32847, 68.38074,
    68.43055, 68.47811, 68.52353, 68.56697, 68.6086, 68.64851, 68.68681, 68.7236,
    68.75903, 68.7931, 68.82594, 68.85757, 68.88815, 68.91765, 68.94617, 68.97377,
    69.0005, 69.02637, 69.05143, 69.07574, 69.09934, 69.12222, 69.14447, 69.1661,
    69.1871, 69.20753, 69.2274, 69.24676, 69.26557, 69.28392, 69.30179, 69.31924,
    69.33621, 69.35278, 69.36894, 69.38474, 69.40012, 69.41514, 69.42981, 69.44412,
    69.45814, 69.47181, 69.48517, 69.49824, 69.51099, 69.5235, 69.53571, 69.54766,
    69.55934, 69.57078, 69.58199, 69.59292, 69.60363, 69.61414, 69.62441, 69.63448,
    69.64435, 69.65398, 69.66344, 69.67271, 69.68179, 69.69065, 69.69939, 69.70792,
    69.71629, 69.72449, 69.73252, 69.74043, 69.74815, 69.75573, 69.76317, 69.77046,
    69.77761, 69.78465, 69.7915, 69.79826, 69.80489, 69.8114, 69.81778, 69.82402,
    69.83018, 69.83621, 69.84214, 69.84796, 69.85366, 69.85923, 69.86477, 69.87016,
    69.87547, 69.88068, 69.88579, 69.8908, 69.89573, 69.9006, 69.90538, 69.91006,
    69.91465, 69.91917, 69.92361, 69.92796
]
# Initial temp. 20degC
T_20 = [
    19.77584, 20.56234, 21.37647, 22.21867, 23.08928, 23.98857, 24.91672, 25.87379,
    26.85972, 27.87432, 28.91727, 29.9881, 31.08615, 32.21057, 33.36039, 34.5344,
    35.73121, 36.94922, 38.18663, 39.44147, 40.71152, 41.99443, 43.28765, 44.58848,
    45.89403, 47.20138, 48.50742, 49.80905, 51.10308, 52.38636, 53.65567, 54.90797,
    56.1402, 57.34949, 58.5331, 59.68847, 60.81322, 61.90522, 62.96259, 63.98366,
    64.96705, 65.91168, 66.81669, 67.6815, 68.50577, 69.28948, 70.03273, 70.73595,
    71.39968, 72.02473, 72.612, 73.16252, 73.67752, 74.15827, 74.60609, 75.02245,
    75.40878, 75.7666, 76.09738, 76.40263, 76.68382, 76.94244, 77.17994, 77.39764,
    77.59691, 77.77908, 77.94529, 78.09682, 78.23475, 78.36013, 78.47397, 78.57718,
    78.67065, 78.75525, 78.83169, 78.90068, 78.96291, 79.01894, 79.06943, 79.11479,
    79.15555, 79.19212, 79.22493, 79.25434, 79.28063, 79.30424, 79.3253, 79.34414,
    79.36096, 79.37594, 79.38934, 79.40127, 79.41189, 79.42137, 79.42979, 79.4373,
    79.44396, 79.44989, 79.45517, 79.45985, 79.464, 79.4677, 79.47098, 79.4739,
    79.47647, 79.47878, 79.48081, 79.4826, 79.48422, 79.48564, 79.48691, 79.48801,
    79.48901, 79.4899, 79.49068, 79.49136, 79.492, 79.49255, 79.49302, 79.49344,
    79.49382, 79.49419, 79.49449, 79.49476, 79.495, 79.49523, 79.49537, 79.49557,
    79.49574, 79.49582, 79.49603, 79.49612, 79.49618, 79.49622, 79.49625, 79.49643,
    79.49657, 79.49658, 79.49659, 79.49661, 79.49661, 79.49664, 79.4967, 79.4967,
    79.49671, 79.49677, 79.4969, 79.49696, 79.49696, 79.49696, 79.49697, 79.49698,
    79.49703, 79.49704, 79.49704, 79.49704, 79.49704, 79.49704, 79.49704, 79.49705,
    79.49706, 79.49706, 79.49706, 79.49706, 79.49706, 79.49706, 79.49706, 79.49706,
    79.49706, 79.49706, 79.49706, 79.49706, 79.49706, 79.49706, 79.49706, 79.49706,
    79.49706, 79.49706, 79.49706, 79.49706
]
# Initial temp. 30degC
T_30 = [
    30.14297, 30.7003, 31.29086, 31.9165, 32.5791, 33.28051, 34.02258, 34.80703,
    35.63555, 36.50961, 37.43052, 38.39928, 39.41662, 40.48286, 41.59785, 42.76102,
    43.97115, 45.22647, 46.52449, 47.86212, 49.23556, 50.64037, 52.07148, 53.5233,
    54.98974, 56.46436, 57.94049, 59.41138, 60.87024, 62.3106, 63.72621, 65.11124,
    66.46048, 67.76934, 69.03381, 70.25064, 71.41739, 72.53218, 73.59389, 74.60198,
    75.55654, 76.45811, 77.30772, 78.10664, 78.85653, 79.55928, 80.21693, 80.83157,
    81.40549, 81.94088, 82.44002, 82.90502, 83.33809, 83.74133, 84.11666, 84.46604,
    84.79128, 85.09399, 85.37583, 85.6383, 85.88275, 86.11049, 86.32278, 86.52075,
    86.70538, 86.87769, 87.03855, 87.1888, 87.32927, 87.46046, 87.58329, 87.6982,
    87.80579, 87.90659, 88.00104, 88.08961, 88.17271, 88.25072, 88.32395, 88.39279,
    88.45748, 88.5183, 88.57554, 88.62943, 88.68014, 88.72795, 88.77299, 88.81546,
    88.85555, 88.89339, 88.92912, 88.96287, 88.99474, 89.02489, 89.05343, 89.08041,
    89.10596, 89.13016, 89.15307, 89.17478, 89.19538, 89.21492, 89.23347, 89.25108,
    89.26778, 89.28367, 89.29875, 89.3131, 89.32677, 89.33977, 89.35213, 89.36391,
    89.37511, 89.3858, 89.39602, 89.40573, 89.41502, 89.42387, 89.4323, 89.44038,
    89.44809, 89.45543, 89.46251, 89.46926, 89.47565, 89.48186, 89.48775, 89.49343,
    89.49879, 89.50402, 89.50899, 89.51377, 89.51832, 89.52273, 89.52689, 89.531,
    89.5349, 89.53857, 89.54218, 89.54566, 89.54894, 89.55213, 89.55525, 89.5582,
    89.56107, 89.56371, 89.56638, 89.56895, 89.57135, 89.57376, 89.576, 89.57816,
    89.58033, 89.58239, 89.58436, 89.58627, 89.58809, 89.58981, 89.59155, 89.5932,
    89.5948, 89.59633, 89.59787, 89.59927, 89.60073, 89.60202, 89.60332, 89.60466,
    89.60582, 89.60701, 89.60813, 89.60928, 89.6104, 89.61147, 89.61246, 89.61343,
    89.6144, 89.61533, 89.6162, 89.61702
]

"""Weight Distributions: Plot histograms of weight values for each layer. This can reveal information about weight initialization, symmetry, and whether weights are sparse or concentrated around certain values."""

import torch
import matplotlib.pyplot as plt

# Instantiate your FCN model
model = FCN(layers)  # Make sure 'layers' is defined before this line

# Create a list to store weight tensors
weights_list = []

# Access the model's state dictionary and extract weight tensors
state_dict = model.state_dict()
for key, value in state_dict.items():
    if "weight" in key:  # Check if the parameter is a weight tensor
        weights_list.append(value)

# Plot histograms of weight distributions
plt.figure(figsize=(12, 6))
for i, weights in enumerate(weights_list):
    plt.subplot(1, len(weights_list), i + 1)
    plt.hist(weights.data.numpy().flatten(), bins=50, color='blue', alpha=0.7)
    plt.title(f'Layer {i + 1} Weights Distribution')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
#plt.title(r'Absolute Error $|T(x,t)- \hat T(x,t)| ℃$', fontsize=18)
plt.tight_layout()
plt.show()

import torch
import matplotlib.pyplot as plt

# Instantiate your FCN model
model = FCN(layers)  # Make sure 'layers' is defined before this line

# Create a list to store weight tensors
weights_list = []

# Access the model's state dictionary and extract weight tensors
state_dict = model.state_dict()
for key, value in state_dict.items():
    if "weight" in key:  # Check if the parameter is a weight tensor
        weights_list.append(value)

# Create heatmaps for weight matrices
plt.figure(figsize=(12, 6))
for i, weights in enumerate(weights_list):
    plt.subplot(1, len(weights_list), i + 1)
    plt.imshow(weights.data.numpy(), cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title(f'Layer {i + 1} Weight Matrix Heatmap')

plt.tight_layout()
plt.show()

#Predicted temperatures:
T10=[ 9.941516, 10.588463, 11.281633, 12.021921, 12.809683, 13.644609,
       14.525664, 15.450942, 16.41766 , 17.42207 , 18.459497, 19.524435,
       20.610611, 21.711205, 22.819115, 23.927258, 25.028908, 26.118206,
       27.190475, 28.242836, 29.274668, 30.288105, 31.288454, 32.284462,
       33.28836 , 34.3154  , 35.382946, 36.508793, 37.709057, 38.99553 ,
       40.37325 , 41.838722, 43.379284, 44.97423 , 46.59698 , 48.21836 ,
       49.809795, 51.346226, 52.807613, 54.179867, 55.45456 , 56.62834 ,
       57.70168 , 58.678066, 59.56285 , 60.362587, 61.084316, 61.735077,
       62.321896, 62.8512  , 63.32916 , 63.761196, 64.15231 , 64.507   ,
       64.8292  , 65.12247 , 65.38993 , 65.63434 , 65.858185, 66.06358 ,
       66.25247 , 66.4266  , 66.58738 , 66.7362  , 66.874176, 67.00246 ,
       67.12187 , 67.2333  , 67.33751 , 67.43512 , 67.52672 , 67.61282 ,
       67.69395 , 67.77051 , 67.842865, 67.911385, 67.97636 , 68.038055,
       68.096756, 68.152695, 68.20607 , 68.25703 , 68.30579 , 68.35247 ,
       68.397255, 68.44025 , 68.481544, 68.52128 , 68.55956 , 68.59645 ,
       68.63203 , 68.66639 , 68.69959 , 68.73171 , 68.76279 , 68.79287 ,
       68.822044, 68.85034 , 68.8778  , 68.90443 , 68.93029 , 68.95549 ,
       68.97992 , 69.00374 , 69.02688 , 69.049446, 69.07138 , 69.09279 ,
       69.11365 , 69.13399 , 69.153824, 69.173164, 69.19204 , 69.21048 ,
       69.228485, 69.24609 , 69.26326 , 69.28002 , 69.29646 , 69.31252 ,
       69.32821 , 69.34355 , 69.35857 , 69.37326 , 69.38766 , 69.40176 ,
       69.41555 , 69.429054, 69.44229 , 69.4553  , 69.46798 , 69.48046 ,
       69.49268 , 69.50465 , 69.51643 , 69.52796 , 69.53927 , 69.55039 ,
       69.5613  , 69.57201 , 69.58254 , 69.592865, 69.60302 , 69.613014,
       69.62283 , 69.63247 , 69.64196 , 69.6513  , 69.66048 , 69.66951 ,
       69.67842 , 69.68719 , 69.695786, 69.7043  , 69.71267 , 69.72094 ,
       69.729065, 69.73708 , 69.74501 , 69.7528  , 69.760506, 69.76812 ,
       69.77562 , 69.783035, 69.79036 , 69.79763 , 69.80476 , 69.81185 ,
       69.818825, 69.82576 , 69.8326  , 69.83936 , 69.84608 , 69.85273 ,
       69.8593  , 69.86584 , 69.87229 , 69.87871 , 69.885056, 69.89138 ]

T20 = [
    19.78, 20.56, 21.38, 22.22, 23.09, 23.99, 24.92, 25.87, 26.86, 27.87, 28.92, 29.99,
    31.09, 32.21, 33.36, 34.53, 35.73, 36.95, 38.19, 39.44, 40.71, 41.99, 43.29, 44.59,
    45.89, 47.20, 48.51, 49.81, 51.10, 52.39, 53.66, 54.91, 56.14, 57.35, 58.53, 59.69,
    60.81, 61.91, 62.96, 63.98, 64.97, 65.91, 66.82, 67.68, 68.51, 69.29, 70.03, 70.74,
    71.40, 72.02, 72.61, 73.16, 73.68, 74.16, 74.61, 75.02, 75.41, 75.77, 76.10, 76.40,
    76.68, 76.94, 77.18, 77.40, 77.60, 77.78, 77.95, 78.10, 78.23, 78.36, 78.47, 78.58,
    78.67, 78.76, 78.83, 78.90, 78.96, 79.02, 79.07, 79.11, 79.16, 79.19, 79.22, 79.25,
    79.28, 79.30, 79.33, 79.34, 79.36, 79.38, 79.39, 79.40, 79.41, 79.42, 79.43, 79.44,
    79.44, 79.45, 79.46, 79.46, 79.46, 79.47, 79.47, 79.47, 79.48, 79.48, 79.48, 79.48,
    79.48, 79.49, 79.49, 79.49, 79.49, 79.49, 79.49, 79.49, 79.49, 79.49, 79.49, 79.49,
    79.49, 79.49, 79.49, 79.50, 79.50, 79.50, 79.50, 79.50, 79.50, 79.50, 79.50, 79.50,
    79.50, 79.50, 79.50, 79.50, 79.50, 79.50, 79.50, 79.50, 79.50, 79.50, 79.50, 79.50,
    79.50, 79.50, 79.50, 79.50, 79.50, 79.50, 79.50, 79.50, 79.50, 79.50, 79.50, 79.50,
    79.50, 79.50, 79.50, 79.50, 79.50, 79.50, 79.50, 79.50, 79.50, 79.50, 79.50, 79.50,
    79.50, 79.50, 79.50, 79.50, 79.50, 79.50, 79.50, 79.50, 79.50, 79.50, 79.50, 79.50
]


T30 = [
    30.14, 30.70, 31.29, 31.92, 32.58, 33.28, 34.02, 34.81, 35.64, 36.51, 37.43, 38.40,
    39.42, 40.48, 41.60, 42.76, 43.97, 45.23, 46.52, 47.86, 49.24, 50.64, 52.07, 53.52,
    54.99, 56.46, 57.94, 59.41, 60.87, 62.31, 63.73, 65.11, 66.46, 67.77, 69.03, 70.25,
    71.42, 72.53, 73.59, 74.60, 75.56, 76.46, 77.31, 78.11, 78.86, 79.56, 80.22, 80.83,
    81.41, 81.94, 82.44, 82.91, 83.34, 83.74, 84.12, 84.47, 84.79, 85.09, 85.38, 85.64,
    85.88, 86.11, 86.32, 86.52, 86.71, 86.88, 87.04, 87.19, 87.33, 87.46, 87.58, 87.70,
    87.81, 87.91, 88.00, 88.09, 88.17, 88.25, 88.32, 88.39, 88.46, 88.52, 88.58, 88.63,
    88.68, 88.73, 88.77, 88.82, 88.86, 88.89, 88.93, 88.96, 88.99, 89.02, 89.05, 89.08,
    89.11, 89.13, 89.15, 89.17, 89.20, 89.21, 89.23, 89.25, 89.27, 89.28, 89.30, 89.31,
    89.33, 89.34, 89.35, 89.36, 89.38, 89.39, 89.40, 89.41, 89.42, 89.42, 89.43, 89.44,
    89.45, 89.46, 89.46, 89.47, 89.48, 89.48, 89.49, 89.49, 89.50, 89.50, 89.51, 89.51,
    89.52, 89.52, 89.53, 89.53, 89.53, 89.54, 89.54, 89.55, 89.55, 89.55, 89.56, 89.56,
    89.56, 89.56, 89.57, 89.57, 89.57, 89.57, 89.58, 89.58, 89.58, 89.58, 89.58, 89.59,
    89.59, 89.59, 89.59, 89.59, 89.59, 89.60, 89.60, 89.60, 89.60, 89.60, 89.60, 89.60,
    89.61, 89.61, 89.61, 89.61, 89.61, 89.61, 89.61, 89.61, 89.61, 89.62, 89.62, 89.62
]

#Reciprocal method:
#Maturity
T_ref1 = 10 #degC equal to 10degC initial temperature
T_ref2 = 20 #degC equal to 20degC initial temperature
T_ref3 = 30 #degC equal to 30degC initial temperature

del_t=180 #hours
M0 = 0 #degC-hours, offset maturity

M1 = [del_t*element- del_t*T_ref1 for element in T10]
M1=np.array(M1)
M1_inverse=1/M1

M2 = [del_t*element- del_t*T_ref2 for element in T20]
M2=np.array(M2)
M2_inverse=1/M2

M3 = [del_t*element- del_t*T_ref3 for element in T30]
M3=np.array(M3)
M3_inverse=1/M3

#Strength
S_inf = 27.4 #MPa 28 days comp. strength
A = 0.28 #Assume regression coefficient
S1_inverse = (1/S_inf)+(1/A)*(1/(M1-M0))
S2_inverse = (1/S_inf)+(1/A)*(1/(M2-M0))
S3_inverse = (1/S_inf)+(1/A)*(1/(M3-M0))
S1 = 1/S1_inverse
S2 = 1/S2_inverse
S3 = 1/S3_inverse
S3

from numpy import savetxt
savetxt('data.csv', S3, delimiter=',') #M1, S1 etc.

# load numpy array from csv file
from numpy import loadtxt
# load array
data = loadtxt('data.csv', delimiter=',')
# print the array
#print(data)
from google.colab import files
files.download("data.csv")

time = list(range(1, 181)) #t 180 hours

# Plot the maturity vs time
plt.plot(time,M1, 'rs-', markersize=4, label='Initial Temp. 10degC')

plt.plot(time,M2, 'bs-', markersize=4, label='Initial Temp. 20degC')

plt.plot(time,M3, 'cs-', markersize=4, label='Initial Temp. 30degC')
plt.legend(loc='upper left')
# Set plot labels and title with increased fontsize
plt.xlabel('Time (hours)', fontsize=15)
plt.ylabel(r'Maturity (°C-hours)', fontsize=15)
plt.xlim(0, 180)
# Increase the size of axis values
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Show the plot
plt.show()
M1.shape

# Plot the strength vs time
plt.plot(time,S1, 'rs-', markersize=4, label='Initial Temp. 10degC')

plt.plot(time,S2, 'bs-', markersize=4, label='Initial Temp. 20degC')

plt.plot(time,S3, 'cs-', markersize=4, label='Initial Temp. 30degC')
plt.legend(loc='upper left')
# Set plot labels and title with increased fontsize
plt.xlabel('Time (hours)', fontsize=15)
plt.ylabel(r'Strength (MPa)', fontsize=15)
plt.xlim(0, 180)
# Increase the size of axis values
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Show the plot
plt.show()

