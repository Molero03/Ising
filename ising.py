import numpy as np
import matplotlib.pyplot as plt
import math as mt
from numba import njit
import imageio
import os



L=32
J=1


ns=10000

s=np.empty([L,L])
for i in range(L):
    
    for j in range(L):
       s[i][j]=np.random.choice([-1,1]) 
    


E=0
for i in range(-1,L-1):
    for j in range(-1,L-1):
        E=E-J*(s[i][j]*s[i+1][j]+s[i][j]*s[i-1][j]+s[i][j]*s[i][j+1]+s[i][j]*s[i][j-1]) 
    


@njit
def average(k ,s, E):

    i=np.random.randint(-1, high=L-1)
    j=np.random.randint(-1, high=L-1)

    dE=2*J*(s[i][j]*s[i+1][j]+s[i][j]*s[i-1][j]+s[i][j]*s[i][j+1]+s[i][j]*s[i][j-1])


    pace=min(1,mt.exp(-k*dE))

    if np.random.random()<pace:
        s[i][j]=-s[i][j]
        E=E+dE


    return E, s


k=4
p=0

plt.xlim(-2,L+1)
plt.ylim(-2,L+1)

plt.pcolormesh(np.arange(L), np.arange(L), s, cmap="gray", edgecolors="none")
plt.gca().set_aspect("equal")
plt.axis("off")  


plt.savefig('frame_{}.png'.format(p), dpi=100)

for t in range(1,ns+1):
        
    E, s =average(k,s,E)
        
    if t%10==0:
        p=p+1
        plt.clf()
        plt.xlim(-2,L+1)
        plt.ylim(-2,L+1)

        plt.pcolormesh(np.arange(L), np.arange(L), s, cmap="gray", edgecolors="none")
        plt.gca().set_aspect("equal")
        plt.axis("off")  


        plt.savefig('frame_{}.png'.format(p), dpi=100)
            

        
        
    
images=[]
n=p
for i in range(n+1):
    images.append(imageio.imread('frame_'+ str(i) + '.png'))

imageio.mimsave('ising.gif', images, duration=1)

for i in range(n+1):
    os.remove('frame_' + str(i) + '.png')


print("GIF created successfully!")
            
