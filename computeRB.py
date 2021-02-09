from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import time
import myHDF5 as myhd

# Solves Poisson in a domain [0,1]x[0,1]
class poissonProblem:
    
    def __init__(self,Nx):    
        # Create mesh and define function space
        self.mesh = UnitSquareMesh(Nx,Nx)
        self.Uh = FunctionSpace(self.mesh, "Lagrange", 1)
        
        # Define Dirichlet boundary (the whole border)
        eps = DOLFIN_EPS
        def boundary(x):
            return np.min(x[0:2]) < eps or np.max(x[0:2]) > 1.0 - eps  
        
        # Define boundary condition
        u0 = Constant(0.0)
        self.bc = DirichletBC(self.Uh, u0, boundary)
        
        # Define variational problem
        self.K11 = Constant(1.0)
        self.K12 = Constant(0.0)
        self.K22 = Constant(1.0)
        self.K = as_matrix(((self.K11,self.K12),(self.K12,self.K22)))
        
        self.uh = TrialFunction(self.Uh) 
        self.vh = TestFunction(self.Uh)
        self.f = Expression("sin(A*x[0])*sin(A*x[1])/(A*A)", degree=2, A = 2*np.pi)
        self.a = inner(self.K*grad(self.uh), grad(self.vh))*dx
        self.b = self.f*self.vh*dx
    
    def solve(self, delta1, delta2):
        self.K11.assign(1.0 + delta1)
        self.K12.assign(delta2)
        self.K22.assign(1.0 - delta1)
        
        # Compute solution
        uh = Function(self.Uh)
        solve(self.a == self.b, uh, self.bc)        
        return uh

Nx = 50

param_limit = [[-0.5,0.5],[-0.25,0.25]]


np.random.seed(9)
Ns = 1000
Np = len(param_limit)

param = np.random.rand(Ns,Np)
for i in range(Np):
    param[:,i] = param_limit[i][0] + (param_limit[i][1] - param_limit[i][0])*param[:,i]  


poisson = poissonProblem(Nx)

S = np.zeros((Ns,poisson.Uh.dim()))

for i in range(Ns):
    start = time.perf_counter()
    print('Solving Snapshot i = ', i)
    S[i,:] = poisson.solve(param[i,0],param[i,1]).vector().get_local()
    end = time.perf_counter()
    print('solved in ', end - start)

plt.figure(1)
plt.scatter(param[:,0],param[:,1])
plt.xlabel('delta 1')
plt.ylabel('delta 2')

myhd.savehd5('train.hd5', S, ['u'], 'w-')
