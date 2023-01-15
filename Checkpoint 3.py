# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 14:25:20 2022

@author: Surface
"""

#imports 
import sys
import numpy as np 
import math
import random 
import matplotlib.pyplot as plt


# =============================================================================
# =============================================================================
# # FUNCTIONS
# =============================================================================
# =============================================================================

#periodic boundary conditions function
def pbc(pos, l):
    # Use mod function to find position of particle within pbc
    return np.mod(pos, l)

def phi(phi_lat, phi0):
    for i in range(N):
        for j in range(N):
            #generate phi value with noise
            phi_lat[i,j] = phi0+np.random.uniform(-0.1,0.1)
    return phi_lat

def mu(mu_lat, phi_lat, a, b, k, del_x, N):
    mu_lat = -a*phi_lat+b*phi_lat**3-(k/del_x**2)*laplacian(phi_lat)
    return mu_lat

def update_phi(phi_lat, mu_lat, M, del_t, del_x, N):
    phi_lat = phi_lat+(M*del_t/del_x**2)*laplacian(mu_lat)
    return phi_lat

def laplacian(lattice):
   return np.roll(lattice, 1, axis=0)+np.roll(lattice, -1, axis=0)+\
       np.roll(lattice, 1, axis=1)+np.roll(lattice, -1, axis=1
                                           )-4*lattice 

def gradient(lattice, dx):
    grady = (np.roll(lattice,-1,axis=0)-np.roll(lattice,1,axis=0))/(2*dx)
    gradx = (np.roll(lattice,-1,axis=1)-np.roll(lattice,1,axis=1))/(2*dx)
    gradz = (np.roll(lattice,-1,axis=2)-np.roll(lattice,1,axis=2))/(2*dx)
    return gradx, grady, gradz

def free_energy(phi_lat, a, k, del_x):
    gradx, grady, gradz = gradient(phi_lat, del_x) 
    f = -0.5*a*(phi_lat)**2+0.25*a*(phi_lat)**4+0.5*k*(gradx+grady+gradz)**2
    return f

def rand_charge(N):
    lattice = np.random.randn(N,N,N)
    return lattice

def point_charge(lattice):
    i = int(len(lattice)/2)
    lattice[i,i,i] = 1
    return lattice

def wire(lattice):
    i = int(len(lattice)/2)
    lattice[:,i,i] = 1
    return lattice

def e_field(pot):
    return -1*np.gradient(pot)

def jacobi3d(lattice, charge, N, dx):
    return(1/6)*(np.roll(lattice,1,axis=0)+np.roll(lattice,-1,axis=0)+
                 np.roll(lattice,1,axis=1)+np.roll(lattice,-1,axis=1)+
                 np.roll(lattice,1,axis=2)+np.roll(lattice,-1,axis=2)+
                 charge*dx**2)                                               
    # for i in range(1,N-1):
    #     for j in range(1,N-1):
    #         for k in range(1,N-1):
    #             lattice[i,j,k] = (1/6)*(lattice[pbc(i+1,N),j,k]+lattice[pbc(i-1,N),j,k]
    #                               +lattice[i,pbc(j+1,N),k]+lattice[i,pbc(j-1,N),k]+
    #                               lattice[i,j,pbc(k+1,N)]+lattice[i,j,pbc(k-1,N)]+
    #                               charge[i,j,k]*dx**2)
    # return lattice

def gauss3d(lattice, lattice_new, charge, N, dx):
    for i in range(1,N-1):
        for j in range(1,N-1):
            for k in range(1,N-1):
                lattice_new[i,j,k] = (1/6)*(lattice_new[i-1,j,k]+lattice[i+1,j,k]
                                           +lattice_new[i,j-1,k]+lattice[i,j+1,k]
                                           +lattice_new[i,j,k-1]+lattice[i,j,k+1]
                                           +charge[i,j,k]*dx**2)
        
    return lattice_new

def gauss_rel(lattice, lattice_new, charge, N, dx, w):
    for i in range(1,N-1):
        for j in range(1,N-1):
            for k in range(1,N-1):
                lattice_new[i,j,k] = lattice[i,j,k]+w*((1/6)*(lattice_new[i-1,j,k]+lattice[i+1,j,k]
                                                               +lattice_new[i,j-1,k]+lattice[i,j+1,k]
                                                               +lattice_new[i,j,k-1]+lattice[i,j,k+1]
                                                               +charge[i,j,k]*dx**2)-lattice[i,j,k])   
    return lattice_new
    

def boundary(lattice):
    lattice[0]=0
    lattice[N-1]=0
    lattice[:,0]=0
    lattice[:,N-1]=0
    lattice[:,:,0]=0
    lattice[:,:,N-1]=0
    return lattice

def boundary_wire(lattice):
    lattice[0]=0
    lattice[N-1]=0
    lattice[:,:,0]=0
    lattice[:,:,N-1]=0
    return lattice


simulation = str(input(">>>simulation (CH/Poisson/SOR/Mag) = "))

# =============================================================================
# Cahn-Hilliard
# =============================================================================

if simulation == "CH":
    steps = int(input(">>>number of steps = ")) 
    N = int(input(">>>lattice dimensions = ")) 
    phi0 = float(input(">>>initial phi = ")) 
    filename = "ch_" + str(phi0) + ".dat"
    del_x = float(input(">>>delta x = "))
    del_t = del_x #0.5*((del_x)**2) von Neumann stability
    animate = input(">>>animate (Y/N) = ") 
        
    M=0.1
    a=0.1
    b=0.1
    k=0.1 
    
    mu_lat = np.zeros((N,N), dtype=float) 
    phi_lat0 = np.zeros((N,N), dtype=float) 
    phi_lat = phi(phi_lat0, phi0)
    
    time = []
    f_energy = []
    
    for t in range(steps): 
        time.append(t)
        new_mu_lat = mu(mu_lat, phi_lat, a, b, k, del_x, N)
        new_phi_lat = update_phi(phi_lat, new_mu_lat, M, del_t, del_x, N)
        mu_lat = new_mu_lat
        phi_lat = new_phi_lat
        f = free_energy(phi_lat, a, k, del_x)
        f_energy.append(np.sum(f))
    
        if(t % 100 == 0): 
            print(t)
            if animate == 'Y':        
                #show animation
                plt.cla()
                im=plt.imshow(phi_lat, animated=True)
                plt.gca().invert_yaxis()
                plt.draw()
                plt.pause(0.0001)
        
    #write data to files
    f = open(filename, 'w')
    for v in range(len(time)):
        f.write(str(time[v]) + ": " + str(f_energy[v]) + "\n")
    f.close()    
                
    plt.title("Free Energy")
    plt.xlabel("Time Step")
    plt.ylabel("Free Energy")
    plt.plot(time, f_energy)
    plt.show()
    
# =============================================================================
# Poisson    
# =============================================================================
    
if simulation == "Poisson":
    
    update = str(input(">>>update type (Jacobi/Gauss) = "))
    if update == "Jacobi":
        filename = "jacobi_pot.dat"
        filename2 = "jacobi_el.dat"
    if update == "Gauss":
        filename = "gauss_pot.dat"
        filename2 = "gauss_el.dat"
        w = float(input(">>>omega = "))
    steps = int(input(">>>number of steps = ")) 
    N = int(input(">>>lattice dimensions = "))+1  
    # del_t = float(input(">>>delta t = ")) 
    dx = 1 # float(input(">>>delta x = "))
    tol = float(input(">>>tolerance = "))
    animate = input(">>>animate (Y/N) = ")

    
    zeros = np.zeros([N,N,N], dtype=float)
    charge = point_charge(zeros)
    # print(charge)
    pot = np.zeros([N,N,N], dtype=float)
    el = zeros
    
    new_pot = np.zeros([N,N,N], dtype=float)
    
    for t in range(steps):
        if(t % 1 == 0):
            print(t)
            if animate == 'Y':        
                    #show animation
                    plt.cla()
                    im=plt.imshow(pot[int(N/2)], animated=True, vmin=0, vmax=0.25)
                    plt.gca().invert_yaxis()
                    plt.colorbar()
                    plt.draw()
                    plt.pause(0.0001)
        if update == "Jacobi": 
            new_pot = jacobi3d(pot, charge, N, dx)
            if np.sum(abs(pot-new_pot))<tol:
                print(t)
                break
            np.copyto(pot, new_pot)
            pot = boundary(pot)
        if update == "Gauss":
            np.copyto(pot, new_pot)
            new_pot = gauss_rel(pot, new_pot, charge, N, dx, w)
            # new_pot = pot+w*(gauss3d(pot, new_pot, charge, N, dx)-pot)
            if np.sum(abs(pot-new_pot))<tol:
                print(t)
                break

            # new_pot = (1-w)*pot+w*gauss3d(pot, new_pot, charge, N, dx)
    
    ex,ey,ez = -1*np.array(np.gradient(pot, edge_order=2))
    norm = np.sqrt(ex**2+ey**2)[:,:,int(N/2)]
    # ex,ey,ez = -1*np.array(gradient(pot, dx))
    # print(el.shape)
    plt.cla()
    plt.quiver(ey[:,:,int(N/2-1)]/norm,ex[:,:,int(N/2-1)]/norm)
    plt.show

    np.savetxt(filename, pot[int(N/2)])
    el = np.array(np.sqrt(ex**2+ey**2)[:,:,int(N/2)])
    # print(el)
    np.savetxt(filename2, el)

                      
# =============================================================================
# SOR
# =============================================================================

if simulation == "SOR":
    
    steps = int(input(">>>number of steps = ")) 
    N = int(input(">>>lattice dimensions = "))+1  
    # del_t = float(input(">>>delta t = ")) 
    dx = 1 # float(input(">>>delta x = "))
    tol = float(input(">>>tolerance = ")) 
       
    w = (1,1.2,1.4,1.6,1.8,2)
    t_list = []
    
    zeros = np.zeros([N,N,N], dtype=float)
    charge = point_charge(zeros)    
    
    for i in range(len(w)):
        # print(charge)
        pot = np.zeros([N,N,N], dtype=float)
        new_pot = np.zeros([N,N,N], dtype=float)
        print(w[i])
        for t in range(steps):
            np.copyto(pot, new_pot)
            new_pot = gauss_rel(pot, new_pot, charge, N, dx, w[i])
            if np.sum(abs(pot-new_pot))<tol or t == steps-1:
                print(t)
                t_list.append(t)
                break
            
    #write data to files
    # f = open('sor.dat','w')
    # for v in range(len(w)):
    #     f.write(str(w[v]) + ": " + str(t_list[v]) + "\n")
    # f.close() 
            
    plt.title("Convergene Time against Omega")
    plt.xlabel("Omega")
    plt.ylabel("Convergence Time")
    plt.plot(w, t_list)
    plt.show()
    
# =============================================================================
# Magnetic Field
# =============================================================================

if simulation == "Mag":
    
    update = str(input(">>>update type (Jacobi/Gauss) = "))
    if update == "Jacobi":
        filename = "jacobi_wpot.dat"
        filename2 = "jacobi_wmag.dat"
    if update == "Gauss":
        filename = "gauss_wpot.dat"
        filename2 = "gauss_wmag.dat"
        w = float(input(">>>omega = "))
    steps = int(input(">>>number of steps = ")) 
    N = int(input(">>>lattice dimensions = "))+1  
    # del_t = float(input(">>>delta t = ")) 
    dx = 1 # float(input(">>>delta x = "))
    tol = float(input(">>>tolerance = "))
    animate = input(">>>animate (Y/N) = ") 
    
    zeros = np.zeros([N,N,N], dtype=float)
    charge = wire(zeros)
    # print(charge)  
    
    pot = np.zeros([N,N,N], dtype=float)  
    new_pot = np.zeros([N,N,N], dtype=float)
    
    for t in range(steps):
        if(t % 100 == 0):
            print(t)
            if animate == 'Y':        
                    #show animation
                    plt.cla()
                    im=plt.imshow(pot[int(N/2)], animated=True, vmin=0, vmax=0.15)
                    plt.gca().invert_yaxis()
                    plt.colorbar()
                    plt.draw()
                    plt.pause(0.0001)
        if update == "Jacobi":    
            new_pot = jacobi3d(pot, charge, N, dx)
            if np.sum(abs(pot-new_pot))<tol:
                print(t)
                break
            np.copyto(pot, new_pot)
            pot = boundary(pot)
        if update == "Gauss":
            np.copyto(pot, new_pot)
            new_pot = gauss_rel(pot, new_pot, charge, N, dx, w)
            # if np.sum(abs(pot-new_pot))<1e-10:
            #     print(t)
            #     break
        
    ex,ey,ez = -1*np.array(np.gradient(pot, edge_order=2))
    mx = ez-ey
    my = ex-ez
    mz = ey-ex    
    norm = np.sqrt(my**2+mz**2)[int(N/2-1),:,:]
    plt.cla()
    plt.quiver(mz[int(N/2-1),:,:]/norm,my[int(N/2-1),:,:]/norm)
    plt.show
    
    np.savetxt(filename, pot[int(N/2-1)])
    mag = np.array(np.sqrt(my**2+mz**2)[int(N/2-1),:,:])
    np.savetxt(filename2, mag)

    