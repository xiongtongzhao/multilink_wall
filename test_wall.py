import dyna9_wall2 as dy
#import dyna9 as dy
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
theta_ini=0
# beta1_ini=0
# beta2_ini=0
# beta3_ini=0
beta1_ini=0
beta2_ini=0
beta3_ini=0
beta4_ini=0
beta5_ini=0
beta6_ini=0
beta7_ini=0
beta8_ini=0
beta9_ini=0

state = np.array([theta_ini,beta1_ini,beta2_ini,beta3_ini,beta4_ini,beta5_ini,beta6_ini,beta7_ini,beta8_ini,beta9_ini],dtype=np.float64)
w=np.zeros((9),dtype=np.float64)
L,e,Yposition,theta,action_absolute,Qu,Q1,Ql,Q2,action=dy.initial(state,w)
A=dy.MatrixA(L,e,Yposition)
print(torch.max(abs(torch.linalg.inv(A))))
B=dy.MatrixB(L,theta,Yposition)
Q=dy.MatrixQ(L,theta,Qu,Q1,Ql,Q2)
C1,C2=dy.MatrixC(action_absolute)
velo=torch.zeros((3),dtype=torch.float64)
velo[0]=1
Cv=torch.matmul(C1,velo.reshape(3,1))+C2

QCv=torch.matmul(Q,Cv)
print(torch.max(abs(QCv)))
# print(QCv)
print(A[0,:])

AQCv=torch.matmul(torch.linalg.inv(A),QCv)
print(AQCv.shape)
CC=AQCv[1::2,:]
# print(torch.max(abs(AQCv)))
BAQCv=torch.matmul(B,AQCv)
x=np.arange(CC.shape[0])*1.0/CC.shape[0]
# plt.scatter(x,CC.view(-1,1))
# plt.show()
print(BAQCv/9)