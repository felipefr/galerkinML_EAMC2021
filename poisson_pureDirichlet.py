#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 14:37:49 2021
Availabel in: https://github.com/felipefr/galerkinML_EAMC2021.git
@author: Felipe Figueredo Rocha, f.rocha.felipe@gmail.com
"""

from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

# Solves Poisson in a domain [0,1]x[0,1]
def solvePoisson(Nx):
    
    # Create mesh and define function space
    mesh = UnitSquareMesh(Nx,Nx)
    Uh = FunctionSpace(mesh, "Lagrange", 1)
    
    # Define Dirichlet boundary (the whole border)
    eps = DOLFIN_EPS
    def boundary(x):
        return np.min(x[0:2]) < eps or np.max(x[0:2]) > 1.0 - eps  
    
    # Define boundary condition
    u0 = Constant(0.0)
    bc = DirichletBC(Uh, u0, boundary)
    
    # Define variational problem
    uh = TrialFunction(Uh) 
    vh = TestFunction(Uh)
    f = Expression("sin(A*x[0])*sin(A*x[1])/(A*A)", degree=2, A = 2*np.pi)
    a = inner(grad(uh), grad(vh))*dx
    b = f*vh*dx
    
    # Compute solution
    uh = Function(Uh)
    solve(a == b, uh, bc)
    
    return uh

uh = solvePoisson(20)
# Plot solution
plt.figure(1)
plot(uh)

mesh = UnitSquareMesh(20,20)
plt.figure(2)
plot(mesh)

plt.show()
