import numpy as np
from os import path
import math
import numpy.matlib
from math import sin
from math import cos
from scipy.sparse.linalg import gmres
from numpy import linalg as LA
import torch

#torch.set_num_threads(4)
N=18
mu=1
device = torch.device('cpu')
N1111=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N1121=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N1122=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N1131=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N1132=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N1141=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N1142=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N1143=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N1151=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N1152=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N1153=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)

N1211=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N1221=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N1222=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N1231=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N1232=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N1241=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N1242=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N1251=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N1252=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)

N1311=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N1321=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N1322=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N1331=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N1332=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N1341=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N1342=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N1343=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)

N1411=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N1421=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N1422=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N1431=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N1432=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N1441=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N1442=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)

N2111=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N2121=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N2122=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N2131=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N2132=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N2141=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N2142=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N2151=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N2152=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)

N2211=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N2221=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N2222=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N2231=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N2232=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N2241=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N2242=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N2243=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N2251=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N2252=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N2253=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)

N2311=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N2321=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N2322=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N2331=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N2332=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N2341=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N2342=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)

N2411=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N2421=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N2422=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N2431=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N2432=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N2441=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N2442=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
N2443=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)

T0_1=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)

T0_3=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
   
 
T1_1=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
    
T1_3=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
      
T2_3=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
     
T3_3=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
   
T0_5=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
    
T1_5=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
    
T2_5=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
    
T3_5=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)
    
T4_5=torch.zeros(((N+1),(N)),dtype=torch.double,device=device)

def cal_remaining_w(x,w):
#     w=np.squeeze(w)
#     x=np.squeeze(x)
    
    A=np.ones((3,9),dtype=np.double)
  
    theta=np.zeros(9,dtype=np.double)
    for i in range(9):
        theta[i]=np.sum(x[1:i+2])
    stheta=np.sin(theta)
    ctheta=np.cos(theta)
    th=x[0]
    b1=x[1]
    b2=x[2]    
    b3=x[3]
    b4=x[4]
    b5=x[5]
    b6=x[6]
    b7=x[7]    
    b8=x[8]
    b9=x[9]    

    
    O1=w[0]
    O2=w[1]
    O3=w[2]    
    O4=w[3]
    O5=w[4]
    O6=w[5]    
    O7=w[6]
    O8=w[7]    
    O9=w[8]
  
    for i in range(9):
        A[1,i]=np.sum(stheta[i:])
        A[2,i]=np.sum(ctheta[i:])    
    AA=np.linalg.pinv(A)
    v=np.dot(np.identity(9)-np.dot(AA,A),w) 
    
    

    #print(np.max(abs(v)))
    if np.max(abs(v))>1:
        v/=np.max(abs(v))
    return v

def Tintegrations(L,e,X1,X2,Yf1,Yf2,Yl1,Yl2):
    global T0_1

    global T0_3
   
 
    global T1_1
    
    global T1_3
      
    global T2_3
     
    global T3_3
   
    global T0_5
    
    global T1_5
    
    global T2_5
    
    global T3_5
    
    global T4_5
    
#     Yf2=-Yff2
#     Yl2=-Yll2    
    R1=torch.sqrt(e**2+(X1-Yl1)**2+(X2-Yl2)**2)

    y1y1=(X1-Yf1)*(X1-Yf1)
    y2y2=(X2-Yf2)*(X2-Yf2)
    
#     y1y1_image=y1y1.copy()
#     y2y2_image=(X2+Yf2)*(X2+Yf2)    
    
    
    R0=torch.sqrt(e**2+y1y1+y2y2)
#     R0_image=torch.sqrt(e**2+y1y1_image+y2y2_image)
    
    y1v1=(X1-Yf1)*(Yf1-Yl1)
    y2v2=(X2-Yf2)*(Yf2-Yl2)
    xv1=y1v1+y2v2+L**2
    B=xv1-L**2
    
    R02=R0**2    
    R13=R1**3
    R13D=torch.reciprocal(R13)
    
    
    xv0=xv1-L**2
    R1D=torch.reciprocal(R1)
    R0D=torch.reciprocal(R0)
    
    T0_1=(torch.log(L*R1+xv1)-torch.log(L*R0+xv0))/L

    T0_3=torch.reciprocal(R0*(L*R0+xv0))-torch.reciprocal(R1*(L*R1+xv1))
   
     
    T1_1=(R1-R0)/(L**2)-xv0*T0_1/(L**2)
    
    T1_3=(R0D-R1D)/(L**2)-xv0*T0_3/(L**2)
      
    T2_3=-R1D/(L*L) + T0_1/(L**2) - xv0*T1_3/(L**2)
     
    T3_3=-R1D/(L*L) + 2*T1_1/(L**2) - xv0*T2_3/(L**2)
   
    #T0_5=(3*(L**2)*(B**2)+6*(L**4)*(B)+3*(R02)*(L**4)+2*(L**6))*R03D/(3*(B**2-(L**2)*R02)**2)
    T0_5= (B+L**2)*(-B**2+L**2*(3*R0**2+4*B+2*L**2))/((3*R1**3)*((B**2-L**2*R0**2)**2))-B*(-B**2+L**2*(3*R0**2))/((3*R0**3)*(B**2-L**2*R0**2)**2)
    
    
    #print(T0_5)
    
    #print(torch.max(T0_6-T0_5))
    T1_5=-B/(L**2)*T0_5-(R13D-1/(R0**3))/(3*(L**2))
    
    T2_5=-1/(3*(L**2))*R13D+1/(3*(L**2))*T0_3-B*T1_5/(L**2)
    
    T3_5=-1/(3*(L**2))*R13D+2/(3*(L**2))*T1_3-B*T2_5/(L**2)
    
    T4_5=-1/(3*(L**2))*R13D+1/((L**2))*T2_3-B*T3_5/(L**2)
    
#     T0_5=  0
#     T1_5=0    
#     T2_5=  0
#     T3_5=0     
#     T4_5=0   
    
    #return [T0_1,T0_3,T1_1,T1_3,T2_3,T3_3,T0_5,T1_5,T2_5,T3_5,T4_5,y1y1,y2y2,y1v1,y2v2]



def N_matrix(L,e,X1,X2,Yf1,Yf2,Yl1,Yl2):
    
    global N1111
    global N1121  
    global N1122
    global N1131
    global N1132   
    global N1141    
    global N1142     
    global N1143
    global N1151    
    global N1152     
    global N1153    
    
    global N1211
    global N1221  
    global N1222
    global N1231
    global N1232    
    global N1241    
    global N1242    
    global N1251    
    global N1252
    
    global N1311    
    global N1321    
    global N1322
    global N1331
    global N1332    
    global N1341    
    global N1342    
    global N1343
    
    global     N1411
    global     N1421
    global     N1422
    global     N1431  
    global     N1432
    global     N1441
    global     N1442
    
    
    global    N2211  
    global    N2221
    global    N2222
    global    N2231
    
    global    N2232
    global    N2241
    global    N2242
    global    N2243    
    global    N2251
    global    N2252 
    global    N2253
    
    global    N2411
    global    N2421
    global    N2422
    global    N2431
    global    N2432
    global    N2441
    global    N2442
    global    N2443     
    
    
    Yf2_image=-Yf2
   
    Yl2_image=-Yl2
    v1=Yf1-Yl1
    v2=Yf2_image-Yl2_image
    Y1=X1-Yf1
    Y2=X2-Yf2_image
    
    N1111=-6*(v1**2)*v2*(Y2+Yf2_image)
    
    N1121=6*(v1)*(Y2+Yf2_image)*(v1*v2+v1*Yf2_image-2*v2*Y1)
   
    N1122=v1*v1
    
    N1131=-6*(Y2+Yf2_image)*(e**2*v2+v1**2*Yf2_image+v2*Y1**2-2*v1*v2*Y1-2*v1*Yf2_image*Y1)
    N1132= -v1**2+2*Y1*v1+2*v2*Yf2_image+2*v2*Y2
    N1141=  6*(Y2+Yf2_image)*(e**2*v2+e**2*Yf2_image+v2*Y1**2+Yf2_image*Y1**2-2*v1*Yf2_image*Y1)
    N1142=  e**2-2*v1*Y1-2*v2*Y2-2*Yf2_image*Y2-2*v2*Yf2_image-2*Yf2_image**2+Y1**2
    N1143= 1   
    N1151=  -6*Yf2_image*(Yf2_image+Y2)*(e**2+Y1**2)
    N1152=   -e**2+2*Yf2_image**2+2*Yf2_image*Y2-Y1**2
    N1153=-1
      
    N1211=   6*(v1)*(v2**2)*(Y2+Yf2_image) 
    N1221=-6 *(v2)*(Y2+Yf2_image)*(v1*v2+v1*Yf2_image-v2*Y1-v1*Y2)
    N1222=-v1*v2
    N1231=-6*(Y2+Yf2_image)*(v2**2*Y1-v1*v2*Yf2_image+v1*v2*Y2+v1*Yf2_image*Y2+v2*Yf2_image*Y1-v2*Y1*Y2)
    N1232=  v1*v2+2*v1*Yf2_image+v1*Y2-v2*Y1
    N1241= 6*(Y2+Yf2_image)*(v1*Yf2_image*Y2+v2*Yf2_image*Y1-v2*Y1*Y2-Yf2_image*Y1*Y2)
    N1242= v2*Y1-v1*Y2-2*v1*Yf2_image+2*Yf2_image*Y1+Y1*Y2
    N1251= 6*Yf2_image*Y1*Y2*(Y2+Yf2_image)
    N1252=-Y1*(2*Yf2_image+Y2)    
    
    N1311=6*(v1**2)*v2*(Y2+Yf2_image)
    N1321= -6*v1*(Y2+Yf2_image) *(v1*Yf2_image-2*v2*Y1)
    N1322=-v1**2
    N1331= 6*(Y2+Yf2_image)*(v2*e**2+v2*Y1**2-2*v1*Y1*Yf2_image)
    N1332=  -2*v2*Yf2_image-2*v1*Y1-2*v2*Y2
    #print(N1332)
    N1341=-6* Yf2_image*(Y2+Yf2_image)*(e**2+Y1**2)
    N1342=-e**2+2*Yf2_image**2+2*Yf2_image*Y2-Y1**2
    N1343=-1
    
    N1411= -6*v1*v2**2*(Y2+Yf2_image)
    N1421=-6*v2*(Y2+Yf2_image)*(v1*Y2-v1*Yf2_image+v2*Y1)
    N1422= v1*v2
    N1431=  6*(Y2+Yf2_image)*(v1*Yf2_image*Y2+v2*Yf2_image*Y1-v2*Y1*Y2)  
    N1432=v2*Y1-v1*Y2-2*v1*Yf2_image
    N1441=6*Yf2_image*Y1*Y2*(Yf2_image+Y2)
    N1442=-Y1*(Y2+2*Yf2_image)
    
       
    N2211=6*v2**3*(Yf2_image+Y2)    
    N2221=-6*v2**2*(Y2+Yf2_image)*(v2+Yf2_image-2*Y2)
    N2222= v2**2
    N2231= 6*v2*(Yf2_image+Y2)*(v2*Yf2_image-2*v2*Y2-2*Yf2_image*Y2+e**2+Y2**2)
    
    N2232=  -v2**2-2* v2* Yf2_image 
    N2241=-6*(Y2+Yf2_image)*(e**2*v2+e**2*Yf2_image+v2*Y2**2+Yf2_image*Y2**2-2*v2*Yf2_image*Y2)
    N2242=    e**2+2*Yf2_image**2+2*Yf2_image*Y2+2*v2*Yf2_image+Y2**2
    N2243=1    
    N2251=    6*Yf2_image*(Yf2_image+Y2)*(e**2+Y2**2)
    N2252=  -e**2-2*Yf2_image**2-2*Yf2_image*Y2-Y2**2   
    N2253=-1
    
    N2411= -6*v2**3*(Yf2_image+Y2)
    N2421=    -6*v2**2*(-Yf2_image**2+Yf2_image*Y2+2*Y2**2)
    N2422=    -v2**2
    N2431= -6*v2* (Yf2_image+Y2)*(e**2+Y2**2-2*Y2*Yf2_image)
    N2432=  2*v2*  Yf2_image
    N2441= N2251.clone()
    N2442=  N2252.clone()
    N2443=-1    
    
 
    
    #return   [N1111 ,N1121,N1122,N1131,N1132,N1141]
    
    
    
    
def M1M2(L,e,Y):
    global T0_1

    global T0_3
   
 
    global T1_1
    
    global T1_3
      
    global T2_3
     
    global T3_3
   
    global T0_5
    
    global T1_5
    
    global T2_5
    
    global T3_5
    
    global T4_5
    global N1111
    global N1121  
    global N1122
    global N1131
    global N1132   
    global N1141    
    global N1142     
    global N1143
    global N1151    
    global N1152     
    global N1153    
    
    global N1211
    global N1221  
    global N1222
    global N1231
    global N1232    
    global N1241    
    global N1242    
    global N1251    
    global N1252
    
    global N1311    
    global N1321    
    global N1322
    global N1331
    global N1332    
    global N1341    
    global N1342    
    global N1343
    
    global     N1411
    global     N1421
    global     N1422
    global     N1431  
    global     N1432
    global     N1441
    global     N1442
    
    
    global    N2211  
    global    N2221
    global    N2222
    global    N2231
    
    global    N2232
    global    N2241
    global    N2242
    global    N2243    
    global    N2251
    global    N2252 
    global    N2253
    
    global    N2411
    global    N2421
    global    N2422
    global    N2431
    global    N2432
    global    N2441
    global    N2442
    global    N2443
    
    M2=torch.zeros(((N+1)*2,(N)*2),dtype=torch.double,device=device)
    M1=torch.zeros(((N+1)*2,(N)*2),dtype=torch.double,device=device)    

#     #print("--- %s seconds ---" % (time.time() - start_time))  
#     #XX=torch.repeat(Y[:,0],N+1)
    XX=Y[:,0].reshape(-1,1).repeat(1,N+1)
    #XX=XX.reshape(-1,N+1)
  
#     YY=torch.repeat(Y[:,1],N+1)
#     YY=YY.reshape(-1,N+1)
    
    YY=Y[:,1].reshape(-1,1).repeat(1,N+1)    
    #print("--- %s seconds ---" % (time.time() - start_time))    
    X1=XX[:,:-1]
    X2=YY[:,:-1]
    
    Yf1=torch.t(XX[:-1,:])
    Yf2=torch.t(YY[:-1,:])    
    Yl1=torch.t(XX[1:,:])
    Yl2=torch.t(YY[1:,:])    
    
    
    
#     y1=Yf1-X1
#     y2=Yf2-X2
#     v1=Yf1-Yl1
#     v2=Yf2-Yl2
       
    Tintegrations(L,e,X1,X2,Yf1,Yf2,Yl1,Yl2)
#     T0_1=T[0]
#     T0_3=T[1]
#     T1_1=T[2]
#     T1_3=T[3]
#     T2_3=T[4]
#     T3_3=T[5]    
#    
#     #print(y1y1.shape)
#     y1v1=T[8]
#     v1v1=v1*v1
#     y1y2=y1*y2
#     y1v2v1y2=y1*v2+v1*y2
#     v1v2=v1*v2
#     v2v2=(L)**2-v1v1
#     
#     y1y1=T[6]
#     y2y2=T[7]    
#     y2v2=T[9]
     
    
    N_matrix(L,e,X1,X2,Yf1,Yf2,Yl1,Yl2)
    
    M1[0:2*N+2:2,0:2*N:2]=N1111*T4_5+N1121*T3_5+N1122*T3_3+N1131*T2_5+N1132*T2_3+N1141*T1_5+N1142*T1_3+T1_1\
         +N1151*T0_5+N1152*T0_3-T0_1
   
    M1[0:2*N+2:2,1:2*N:2]=N1211*T4_5+N1221*T3_5+N1222*T3_3+N1231*T2_5+N1232*T2_3+N1241*T1_5+N1242*T1_3\
         +N1251*T0_5+N1252*T0_3    
    
    
    M1[1:2*N+2:2,0:2*N:2]=-N1211*T4_5-N1221*T3_5+N1222*T3_3-N1231*T2_5+N1232*T2_3-N1241*T1_5+N1242*T1_3\
         -N1251*T0_5+N1252*T0_3
    
    M1[1:2*N+2:2,1:2*N:2]=N2211*T4_5+N2221*T3_5+N2222*T3_3+N2231*T2_5+N2232*T2_3+N2241*T1_5+N2242*T1_3+T1_1\
         +N2251*T0_5+N2252*T0_3-T0_1
    

    M2[0:2*N+2:2,0:2*N:2]=N1311*T4_5+N1321*T3_5+N1322*T3_3+N1331*T2_5+N1332*T2_3+N1341*T1_5+N1342*T1_3-T1_1
      
    M2[0:2*N+2:2,1:2*N:2]=N1411*T4_5+N1421*T3_5+N1422*T3_3+N1431*T2_5+N1432*T2_3+N1441*T1_5+N1442*T1_3

    M2[1:2*N+2:2,0:2*N:2]=-N1411*T4_5-N1421*T3_5+N1422*T3_3-N1431*T2_5+N1432*T2_3-N1441*T1_5+N1442*T1_3


    M2[1:2*N+2:2,1:2*N:2]= N2411*T4_5+N2421*T3_5+N2422*T3_3+N2431*T2_5+N2432*T2_3+N2441*T1_5+N2442*T1_3-T1_1   
    #print(N1332)
        
    
#     M2[0:2*N+2:2,0:2*N:2]=L*((T1_1+e**2*T1_3)+T1_3*y1y1+T2_3*(y1v1*2)+T3_3*v1v1)
# 
#     
#     M2[0:2*N+2:2,1:2*N:2]=L*(T1_3*y1y2+T2_3*(y1v2v1y2)+T3_3*v1v2)
# 
#     
#     M2[1:2*N+2:2,0:2*N:2]=M2[0:2*N+2:2,1:2*N:2]
# 
#     
#     M2[1:2*N+2:2,1:2*N:2]=L*((T1_1+e**2*T1_3)+T1_3*y2y2+T2_3*(y2v2*2)+T3_3*v2v2)
#     
#     
#     M1[1:2*N+2:2,1:2*N:2]=L*((T0_1+e**2*T0_3)+T0_3*y2y2+T1_3*(y2v2*2)+T2_3*v2v2)- M2[1:2*N+2:2,1:2*N:2]    
    
    #print(M2[2*N+1,2*N-1],5)
    #print((T1_1+e**2*T1_3).shape,(T1_3*torch.multiply(y2,y2)).shape,(T2_3*(torch.multiply(y2,v2)+torch.multiply(v2,y2)))[-1,-1],(T3_3*torch.multiply(v2,v2))[-1,-1])
    
    
    
    
#     M1[0:2*N+2:2,0:2*N:2]=L*((T0_1+e**2*T0_3)+T0_3*y1y1+T1_3*(y1v1*2)+T2_3*v1v1)-M2[0:2*N+2:2,0:2*N:2]
#     M1[0:2*N+2:2,1:2*N:2]=L*(T0_3*y1y2+T1_3*(y1v2v1y2)+T2_3*v1v2)-M2[0:2*N+2:2,1:2*N:2]
#     M1[1:2*N+2:2,0:2*N:2]=M1[0:2*N+2:2,1:2*N:2]
    
    #M1[1:2*N+2:2,1:2*N:2]=L*((T0_1+e**2*T0_3)+T0_3*y2y2+T1_3*(y2v2*2)+T2_3*v2v2)- M2[1:2*N+2:2,1:2*N:2]
    
    M1=torch.cat((M1,torch.zeros(((N+1)*2,2),dtype=torch.double,device=device)),dim=1)
    M2=torch.cat((torch.zeros(((N+1)*2,2),dtype=torch.double,device=device),M2),dim=1)
    
  
    return M1, M2
    

def MatrixA(L,e,Y):  

    M1,M2=M1M2(L,e,Y)
  
    A=(M1+M2)/L         
    #print(M2*L)
    return A/(8*math.pi*mu)    



def MatrixQ(L,theta,Qu,Q1,Ql,Q2):

    
    Q=torch.cat((Qu,-Q2,Ql,Q1),dim=1)    
    Q=Q.reshape(2*(N+1),-1)

    return Q


def MatrixQp(L,theta):
    
    Qu=torch.cat((torch.ones((N+1),dtype=torch.double,device=device).reshape(-1,1),torch.zeros((N+1),dtype=torch.double,device=device).reshape(-1,1)),dim=1)
    Ql=torch.cat((torch.zeros((N+1),dtype=torch.double,device=device).reshape(-1,1),torch.ones((N+1),dtype=torch.double,device=device).reshape(-1,1)),dim=1)
    q1=L*torch.cos(theta[2:])
    q2=L*torch.sin(theta[2:])
   
    

    Q1=q1.reshape(1,-1).repeat(N+1,1)
    
    Q1=torch.tril(Q1,-1)
    
    Q2=q2.reshape(1,-1).repeat(N+1,1)
    
    Q2=torch.tril(Q2,-1)    
    

    
    Q=torch.cat((Qu,Q1,Ql,Q2),dim=1) 
    Q=Q.reshape(2*(N+1),-1)

    return Q,Qu,Q1,Ql,Q2


def MatrixB(L,theta,Y):
    
    B1=0.5*L*torch.cat((2*torch.ones((N+1),dtype=torch.double,device=device).reshape(-1,1),torch.zeros((N+1),dtype=torch.double,device=device).reshape(-1,1)),dim=1)
    B1[0,0]=0.5*L
    B1[-1,0]=0.5*L
    B1=B1.reshape(1,-1)
#     
    B2=0.5*L*torch.cat((torch.zeros((N+1),dtype=torch.double,device=device).reshape(-1,1),2*torch.ones((N+1),dtype=torch.double,device=device).reshape(-1,1)),dim=1)
    B2[0,1]=0.5*L
    B2[-1,1]=0.5*L
    B2=B2.reshape(1,-1)
    
    Y1=torch.cat((Y[:-1,:],torch.zeros((2),dtype=torch.double,device=device).reshape(1,-1)),dim=0)
    
    Y2=torch.cat((torch.zeros((2),dtype=torch.double,device=device).reshape(1,-1),Y[:-1,:]),dim=0)

    
#     Y01=torch.repeat(Y[0,:],N+1)
#     Y01=Y01.reshape(-1,N+1).T
    #print(Y.shape)
    Y01=Y[0,:].reshape(1,-1).repeat(N+1,1)
    #Y02=Y01.detach()    
    Y01[-1,0]=0
    Y01[-1,1]=0
# torch.repeat(Y[:,0],N+1) XX.reshape(-1,N+1)

#     Y02=torch.repeat(Y[0,:],N+1)
#     Y02=Y02.reshape(-1,N+1).T
    Y02=Y[0,:].reshape(1,-1).repeat(N+1,1)
    Y02[0,0]=0
    Y02[0,1]=0
    #np.savetxt('Y02.out', Y02.numpy(), delimiter=',')
    t=torch.cat((torch.cos(theta[2:]).reshape(-1,1),torch.sin(theta[2:]).reshape(-1,1)),dim=1)
     
    t1= torch.cat((t,torch.zeros((2),dtype=torch.double,device=device).reshape(1,-1)),dim=0)
   
    t2= torch.cat((torch.zeros((2),dtype=torch.double,device=device).reshape(1,-1),t),dim=0)
    B3=0.5*L*(Y1-Y01)+(L**2)/6.0*t1+0.5*L*(Y2-Y02)+(L**2)/3.0*t2
    #np.savetxt('B3.out', B3.numpy(), delimiter=',')  
    B3=torch.cat((-B3[:,1].reshape(-1,1),B3[:,0].reshape(-1,1)),dim=1)
    #B3=B3.reshape(2,-1).T
    B3=B3.reshape(1,-1)
    
    #np.savetxt('B3.out', B3.numpy(), delimiter=',')     
    B=torch.cat((B1,B2,B3),dim=0)
    #print(torch.mean(B1),torch.mean(B2),torch.mean(B3),'B')  
    return B

def MatrixC(action_absolute):
    C1=torch.zeros((N+2,3),dtype=torch.double,device=device)
    C1[0,0]=1
    C1[1,1]=1
    C1[2:,2]=1
    C2=torch.zeros((N+2,1),dtype=torch.double,device=device)
    C2[3:,:]=action_absolute.view(-1,1) #N-1,1, start's rotation velocity removed
    #print(C1,C2)
    return C1, C2



def Calculate_velocity(x,w):
    L,e,Y,theta,action_absolute,Qu,Q1,Ql,Q2,action=initial(x,w)    
    B=MatrixB(L,theta,Y)
    
    A=MatrixA(L,e,Y)
  
    Q=MatrixQ(L,theta,Qu,Q1,Ql,Q2)

    C1,C2=MatrixC(action_absolute)

    AB=torch.zeros((3,A.shape[0]),dtype=torch.double,device=device)
    
    AB = torch.linalg.solve(A.T, B.T)
        
        
    MT=torch.matmul((AB.T).double(),Q)     
    
        
       
    M=torch.matmul(MT,C1)

    R=-torch.matmul(MT,C2)
    
    velo=torch.matmul(torch.linalg.inv(M),R).numpy()   
    velo=np.squeeze(velo)
    omega=velo[2]    
    O1=action[0]
    O2=action[1]
    O3=action[2]
    O4=action[3]
    O5=action[4]
    O6=action[5]    
    O7=action[6]    
    O8=action[7]
    O9=action[8]    
   
    u1=velo[0]-0.5*omega*sin(x[0])
    v1=velo[1]+0.5*omega*cos(x[0])
    
    u2=velo[0]-omega*sin(x[0])-0.5*(omega+O1)*sin(x[0]+x[1])
    v2=velo[1]+omega*cos(x[0])+0.5*(omega+O1)*cos(x[0]+x[1])
    
    u3=velo[0]-omega*sin(x[0])-(omega+O1)*sin(x[0]+x[1])-0.5*(omega+O1+O2)*sin(x[0]+x[1]+x[2])
    v3=velo[1]+omega*cos(x[0])+(omega+O1)*cos(x[0]+x[1])+0.5*(omega+O1+O2)*cos(x[0]+x[1]+x[2])
    
    u4=velo[0]-omega*sin(x[0])-(omega+O1)*sin(x[0]+x[1])-(omega+O1+O2)*sin(x[0]+x[1]+x[2])-0.5*(omega+O1+O2+O3)*sin(x[0]+x[1]+x[2]+x[3])
    v4=velo[1]+omega*cos(x[0])+(omega+O1)*cos(x[0]+x[1])+(omega+O1+O2)*cos(x[0]+x[1]+x[2])+0.5*(omega+O1+O2+O3)*cos(x[0]+x[1]+x[2]+x[3])
    
    u5=velo[0]-omega*sin(x[0])-(omega+O1)*sin(x[0]+x[1])-(omega+O1+O2)*sin(x[0]+x[1]+x[2])-(omega+O1+O2+O3)*sin(x[0]+x[1]+x[2]+x[3])\
        -0.5*(omega+O1+O2+O3+O4)*sin(x[0]+x[1]+x[2]+x[3]+x[4])
        
        
    v5=velo[1]+omega*cos(x[0])+(omega+O1)*cos(x[0]+x[1])+(omega+O1+O2)*cos(x[0]+x[1]+x[2])+(omega+O1+O2+O3)*cos(x[0]+x[1]+x[2]+x[3])\
        +0.5*(omega+O1+O2+O3+O4)*cos(x[0]+x[1]+x[2]+x[3]+x[4])
    
    u6=velo[0]-omega*sin(x[0])-(omega+O1)*sin(x[0]+x[1])-(omega+O1+O2)*sin(x[0]+x[1]+x[2])-(omega+O1+O2+O3)*sin(x[0]+x[1]+x[2]+x[3])\
        -(omega+O1+O2+O3+O4)*sin(x[0]+x[1]+x[2]+x[3]+x[4])-0.5*(omega+O1+O2+O3+O4+O5)*sin(x[0]+x[1]+x[2]+x[3]+x[4]+x[5])
        
        
    v6=velo[1]+omega*cos(x[0])+(omega+O1)*cos(x[0]+x[1])+(omega+O1+O2)*cos(x[0]+x[1]+x[2])+(omega+O1+O2+O3)*cos(x[0]+x[1]+x[2]+x[3])\
        +(omega+O1+O2+O3+O4)*cos(x[0]+x[1]+x[2]+x[3]+x[4]) +   0.5*(omega+O1+O2+O3+O4+O5)*cos(x[0]+x[1]+x[2]+x[3]+x[4]+x[5])
    
    
    
    u7=velo[0]-omega*sin(x[0])-(omega+O1)*sin(x[0]+x[1])-(omega+O1+O2)*sin(x[0]+x[1]+x[2])-(omega+O1+O2+O3)*sin(x[0]+x[1]+x[2]+x[3])\
        -(omega+O1+O2+O3+O4)*sin(x[0]+x[1]+x[2]+x[3]+x[4])-(omega+O1+O2+O3+O4+O5)*sin(x[0]+x[1]+x[2]+x[3]+x[4]+x[5]) -\
        0.5*(omega+O1+O2+O3+O4+O5+O6)*sin(x[0]+x[1]+x[2]+x[3]+x[4]+x[5]+x[6])        
    
    v7=velo[1]+omega*cos(x[0])+(omega+O1)*cos(x[0]+x[1])+(omega+O1+O2)*cos(x[0]+x[1]+x[2])+(omega+O1+O2+O3)*cos(x[0]+x[1]+x[2]+x[3])\
        +(omega+O1+O2+O3+O4)*cos(x[0]+x[1]+x[2]+x[3]+x[4]) +   (omega+O1+O2+O3+O4+O5)*cos(x[0]+x[1]+x[2]+x[3]+x[4]+x[5])  +\
        0.5*(omega+O1+O2+O3+O4+O5+O6)*cos(x[0]+x[1]+x[2]+x[3]+x[4]+x[5]+x[6])
    
    
    u8=velo[0]-omega*sin(x[0])-(omega+O1)*sin(x[0]+x[1])-(omega+O1+O2)*sin(x[0]+x[1]+x[2])-(omega+O1+O2+O3)*sin(x[0]+x[1]+x[2]+x[3])\
        -(omega+O1+O2+O3+O4)*sin(x[0]+x[1]+x[2]+x[3]+x[4])-(omega+O1+O2+O3+O4+O5)*sin(x[0]+x[1]+x[2]+x[3]+x[4]+x[5]) -\
        (omega+O1+O2+O3+O4+O5+O6)*sin(x[0]+x[1]+x[2]+x[3]+x[4]+x[5]+x[6]) - 0.5*(omega+O1+O2+O3+O4+O5+O6+O7)*sin(x[0]+x[1]+x[2]+x[3]+x[4]+x[5]+x[6]+x[7])        
    
    v8=velo[1]+omega*cos(x[0])+(omega+O1)*cos(x[0]+x[1])+(omega+O1+O2)*cos(x[0]+x[1]+x[2])+(omega+O1+O2+O3)*cos(x[0]+x[1]+x[2]+x[3])\
        +(omega+O1+O2+O3+O4)*cos(x[0]+x[1]+x[2]+x[3]+x[4]) +   (omega+O1+O2+O3+O4+O5)*cos(x[0]+x[1]+x[2]+x[3]+x[4]+x[5])  +\
        (omega+O1+O2+O3+O4+O5+O6)*cos(x[0]+x[1]+x[2]+x[3]+x[4]+x[5]+x[6])+ 0.5*(omega+O1+O2+O3+O4+O5+O6+O7)*cos(x[0]+x[1]+x[2]+x[3]+x[4]+x[5]+x[6]+x[7]) 


    u9=velo[0]-omega*sin(x[0])-(omega+O1)*sin(x[0]+x[1])-(omega+O1+O2)*sin(x[0]+x[1]+x[2])-(omega+O1+O2+O3)*sin(x[0]+x[1]+x[2]+x[3])\
        -(omega+O1+O2+O3+O4)*sin(x[0]+x[1]+x[2]+x[3]+x[4])-(omega+O1+O2+O3+O4+O5)*sin(x[0]+x[1]+x[2]+x[3]+x[4]+x[5]) -\
        (omega+O1+O2+O3+O4+O5+O6)*sin(x[0]+x[1]+x[2]+x[3]+x[4]+x[5]+x[6]) - (omega+O1+O2+O3+O4+O5+O6+O7)*sin(x[0]+x[1]+x[2]+x[3]+x[4]+x[5]+x[6]+x[7])\
        -0.5*(omega+O1+O2+O3+O4+O5+O6+O7+O8)*sin(x[0]+x[1]+x[2]+x[3]+x[4]+x[5]+x[6]+x[7]+x[8])
    
    v9=velo[1]+omega*cos(x[0])+(omega+O1)*cos(x[0]+x[1])+(omega+O1+O2)*cos(x[0]+x[1]+x[2])+(omega+O1+O2+O3)*cos(x[0]+x[1]+x[2]+x[3])\
        +(omega+O1+O2+O3+O4)*cos(x[0]+x[1]+x[2]+x[3]+x[4]) +   (omega+O1+O2+O3+O4+O5)*cos(x[0]+x[1]+x[2]+x[3]+x[4]+x[5])  +\
        (omega+O1+O2+O3+O4+O5+O6)*cos(x[0]+x[1]+x[2]+x[3]+x[4]+x[5]+x[6])+ (omega+O1+O2+O3+O4+O5+O6+O7)*cos(x[0]+x[1]+x[2]+x[3]+x[4]+x[5]+x[6]+x[7])\
        +0.5*(omega+O1+O2+O3+O4+O5+O6+O7+O8)*cos(x[0]+x[1]+x[2]+x[3]+x[4]+x[5]+x[6]+x[7]+x[8])


    #print(velo)
    u=(u1+u2+u3+u4+u5+u6+u7+u8+u9)/9
    v=(v1+v2+v3+v4+v5+v6+v7+v8+v9)/9


    print(velo,'here')    
    
    
    return np.array([u.item(),v.item(),omega.item(),O1,O2,O3,O4,O5,O6,O7,O8,O9])




def initial(x,w):




    N1=int(N/9)-1
    N2=int(N/9)*2-1
    N3=int(N/9)*3-1
    N4=int(N/9)*4-1
    N5=int(N/9)*5-1
    N6=int(N/9)*6-1    
    N7=int(N/9)*7-1
    N8=int(N/9)*8-1
    N9=int(N/9)*9-1 

    #L=2.0/N
    L=9.0/N
   
    e=0.09
    
    Xini=0
    Yini=10
    # beta1_ini=math.pi*0.0
    # beta2_ini=math.pi/2
    # beta3_ini=math.pi
    # beta4_ini=math.pi*1.5


     
    beta_ini=torch.tensor(x[:-1].copy(),dtype=torch.double,device=device)
    beta_ini[1]+=beta_ini[0]
    beta_ini[2]+=beta_ini[1]
    beta_ini[3]+=beta_ini[2]
    beta_ini[4]+=beta_ini[3]
    beta_ini[5]+=beta_ini[4]
    beta_ini[6]+=beta_ini[5]
    beta_ini[7]+=beta_ini[6]
    beta_ini[8]+=beta_ini[7]    
    
   
    # beta1_ini=0
    # beta2_ini=0
    # beta3_ini=0
    # beta4_ini=0
    theta=torch.zeros((N+2),dtype=torch.double,device=device)
    forQp=torch.ones((N+2),dtype=torch.double,device=device)
    forQp[0]=Xini
    forQp[1]=Yini

    theta[0]=Xini
    theta[1]=Yini

    for i in range(N):
        theta[i+2]=beta_ini[int((i)/(N/9))]    


   
    Q,Qu,Q1,Ql,Q2=MatrixQp(L,theta)
    Yposition=torch.matmul(Q,forQp)            


    
    Yposition=Yposition.reshape(-1,2)
    #print(Yposition,theta)

    absU=cal_remaining_w(x,w)
    action=absU.copy() 
    absU[1]=absU[1]+absU[0]
    absU[2]=absU[2]+absU[1]
    absU[3]=absU[3]+absU[2]
    absU[4]=absU[4]+absU[3]
    absU[5]=absU[5]+absU[4]
    absU[6]=absU[6]+absU[5]
    absU[7]=absU[7]+absU[6]
    #absU[8]=absU[8]+absU[7]    
    

    


    
    action_absolute=torch.zeros((N-1),dtype=torch.double,device=device)
    action_absolute[N1:N2]=absU[0]
    action_absolute[N2:N3]=absU[1]
    action_absolute[N3:N4]=absU[2]
    action_absolute[N4:N5]=absU[3]
    action_absolute[N5:N6]=absU[4]
    action_absolute[N6:N7]=absU[5]  
    action_absolute[N7:N8]=absU[6]
    action_absolute[N8:]=absU[7]
    
    action_absolute.view(-1,1)
    return L,e,Yposition,theta,action_absolute,Qu,Q1,Ql,Q2,action



def RK(x,w):
    Xn=0.0
    Yn=0.0
    r=0.0
    xc=x+1.0
    xc=xc-1.0
    for i in range(5):
        #print(xc.shape,w.shape)
        V=Calculate_velocity(xc, w)
        k2=0.01*V
        
        #k2=0.01*Calculate_velocity(xc+0.5*k1[2:], w)
        xc+=k2[2:]
        #xc=(xc+math.pi)%(2*math.pi)-math.pi
          
        Xn+=k2[0]
        Yn+=k2[1]
        r+=(k2[0])/(0.05)    

    return xc , Xn, Yn ,r  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
