import numpy as np
import matplotlib.pyplot as plt
import os
import time

import multiprocessing as mp
from multiprocessing.sharedctypes import Array
import ctypes

from scipy.optimize import minimize

def naca4digit_gen(M, P, T, n=100): #http://airfoiltools.com/airfoil/naca4digit
    beta = np.linspace(0, np.pi, n)
    x = (1-np.cos(beta))/2 #cosine spacing for better lead edge definition

    y_t = 5*T*(0.2969*np.sqrt(x) - 0.126*x - 0.3516*x**2 + 0.2843*x**3 - 0.1036*x**4)
    y_c = np.zeros_like(x)
    dy_dx = np.zeros_like(x)

    for i, xi in enumerate(x):
        if (0 <= xi < P):
            y_c[i] = M/(P**2) * (2*P*xi - xi**2)
            dy_dx[i] = 2*M /P**2 * (P - xi)
        elif (P <= xi <= 1):
            y_c[i] = M/(1-P)**2 * (1 - 2*P + 2*P*xi - xi**2)
            dy_dx[i] = 2*M /(1-P)**2 * (P - xi)

    theta = np.atan(dy_dx)
    #upper and lower
    x_u = x - y_t * np.sin(theta)
    y_u = y_c + y_t * np.cos(theta)
    
    x_l = x + y_t * np.sin(theta)
    y_l = y_c - y_t * np.cos(theta)

    upper = np.vstack([x_u, y_u]).T
    lower = np.vstack([x_l, y_l]).T
    return upper, lower

def gen_airfoil(x):
    airfoil_upper, airfoil_lower = naca4digit_gen(x[0], x[1], x[2])
    airfoil = np.vstack([airfoil_upper[::-1], airfoil_lower])
    np.savetxt("airfoil_data.csv", airfoil)

def run_xfoil():
    # Knowns
    AoA        = '10'
    writefile = 'output.csv'
    cpfile = 'cps.csv'
    airfoilfile = 'airfoil_data.csv'
    xfoilFlnm  = 'xfoil_input.txt'

    if os.path.exists(writefile):
        os.remove(writefile)
        
    # Create the airfoil
    fid = open(xfoilFlnm,"w")
    fid.write(f"LOAD {airfoilfile}\nfoil\n")
    # fid.write("PPAR\n N 200\n\n\n")
    fid.write("OPER\n")
    # fid.write("VISC 3e6\n")
    # fid.write("M 0.1\n")
    fid.write(f'Pacc\n{writefile}\n\n')
    fid.write(f"ALFA {AoA}\n")
    fid.write(f"CPWR {cpfile}\n")
    fid.write("\nq")
    fid.close()

    # Run the XFoil calling command
    os.system("xfoil < xfoil_input.txt > msgs.csv")

    # Load the data from xfoil results
    dataBuffer = np.loadtxt(writefile, skiprows=12)
    # print(dataBuffer)
    cpBuffer = np.loadtxt(cpfile)

    # Extract data from the loaded dataBuffer array
    C_L = dataBuffer[1]
    C_P = cpBuffer[:,1]
    # print(C_L)
    # print(dataBuffer.shape)

    # if os.path.exists(cpfile):
    #     os.remove(cpfile)
    return C_L, C_P

def run_process(x, target_C_l = 2):
    row = vars[x,:]
    gen_airfoil(row)
    C_l, C_p_list = run_xfoil()
    obj = (C_l - target_C_l)**2 #+ 0.1*np.sum((C_p_list + 1)**2) #to minimize
    output_cls[x] = C_l
    output_objs[x] = obj
    # print(vars[x,:])
    return C_l, obj
 
def obj_function(x, target_C_l=2):
    gen_airfoil(x)
    C_l, C_p_list = run_xfoil()
    obj = (C_l - target_C_l)**2 #+ 0.1*np.sum((C_p_list + 1)**2) #to minimize
    # print(obj)
    row = x[0], x[1], x[2], C_l, obj
    path_points.append(row)
    print(row)
    return obj

def to_numpyarray(mp_arr):
    return np.frombuffer(mp_arr.get_obj(), dtype=np.float32)

def solution_space(build, resolution, n_pts, path_to_sol = None):
    fig = plt.figure(figsize=[60,20])
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')

    if build:
        global n_points
        resolution = resolution
        n_points = n_pts

        m_vals = np.linspace(0, 0.095, resolution)
        p_vals = np.linspace(0.2, 0.9, resolution)
        t_vals = np.linspace(0.10, 0.40, resolution)
        n = resolution**3

        global vars
        m,p,t = np.meshgrid(m_vals, p_vals, t_vals, indexing='ij')
        vars = np.column_stack((m.ravel(), p.ravel(), t.ravel()))
        print(vars.shape)

        global output_cls
        output_cls = Array(ctypes.c_float, n, lock=True)
        global output_objs
        output_objs = Array(ctypes.c_float, n, lock=True)
        iterator = range(int(n))

        # NUMCORES = int(mp.cpu_count())
        NUMCORES = 1
        avg = 0.9
        ops_per_core = n/NUMCORES
        print(f"Each core will process approx {ops_per_core} airfoils")
        print(f"Each airfoil process takes approx {avg}s to complete. Expect a total of {ops_per_core*avg} seconds.")

        tick = time.time()
        with mp.Pool(processes=NUMCORES) as p:
            p.map(run_process, iterator)
        tock = time.time()

        result_cls = to_numpyarray(output_cls)
        result_objs = to_numpyarray(output_objs)
        res = np.column_stack([vars, result_cls, result_objs])
        print(res.shape)
        print(f"Total time elapsed:{(tock-tick):.2f} seconds")

        np.savetxt(f"results/maps/result_data_res{resolution}.csv", res, header='m,p,t,C_l,obj', delimiter=',')
    
    else: 
        res = np.loadtxt(f'results/maps/result_data_res{resolution}.csv', delimiter=',')
        print(res.shape)
        solution = np.array(path_to_sol)
        # ax1.plot(solution[:,0], solution[:,1], solution[:,4], color ='red')
        # ax2.plot(solution[:,1], solution[:,2], solution[:,4], color = 'red')
        # ax3.plot(solution[:,0], solution[:,2], solution[:,4], color = 'red')

    ax1.scatter(solution[-1,0], solution[-1,1], solution[-1,4], color ='green')
    ax2.scatter(solution[-1,1], solution[-1,2], solution[-1,4], color = 'green')
    ax3.scatter(solution[-1,0], solution[-1,2], solution[-1,4], color = 'green')

    X = np.linspace(0.283, 0.6166, 50)
    Y_pl = np.sqrt(1-(6*(X-0.45))**2)*1/12 + 0.2
    Y_min = -np.sqrt(1-(6*(X-0.45))**2)*1/12 + 0.2
    ax2.plot(np.append(X,X), np.append(Y_pl, Y_min), color = 'orange')

    ax1.scatter(solution[0,0], solution[0,1], solution[0,4], color ='red')
    ax2.scatter(solution[0,1], solution[0,2], solution[0,4], color = 'red')
    ax3.scatter(solution[0,0], solution[0,2], solution[0,4], color = 'red')
    # print(res[:,0])
    header = ['m', 'p', 't', 'C_l', 'obj']
    X = res[:,0]
    Y = res[:,1]
    Z = res[:,2]
    cls = res[:,3]
    objs = res[:,4]

    print(cls.shape)
    
    ax1.plot_trisurf(X,Y,objs, alpha=0.6)
    ax1.set_xlabel(header[0])
    ax1.set_ylabel(header[1])
    ax1.set_zlabel(header[4])
    # ax1.set_zlim(-1,5)
    
    ax2.plot_trisurf(Y,Z,objs, alpha=0.6)
    ax2.set_xlabel(header[1])
    ax2.set_ylabel(header[2])
    ax2.set_zlabel(header[4])
    # ax2.set_zlim(-1,5)
    
    ax3.plot_trisurf(X,Z,objs, alpha=0.6)
    ax3.set_xlabel(header[0])
    ax3.set_ylabel(header[2])
    ax3.set_zlabel(header[4])
    # ax3.set_zlim(-1,5)
    plt.show()

def optimize(x):
    #start param
    epsilon = 1e-3
    eta = 0.001
    m = [0, 0.095]
    p = [0, 0.9]
    t = [0.01, 0.40]
    for k in range(40):
        CL, J = obj_function(x, 1)
        grad = np.zeros(3)

        for i in range(3):
            x_pert = x.copy()
            x_pert[i] += epsilon
            _,Jp = obj_function(x_pert, 1)
            grad[i] = (Jp - J)/epsilon

        x = x - eta*grad
        x[0] = max(m[0], min(x[0], m[1]))
        x[1] = max(p[0], min(x[1], p[1]))
        x[2] = max(t[0], min(x[2], t[1]))

        print(f"Iter {k}: J={J}, Jp={Jp} m={x[0]:.4f}, p={x[1]:.4f}, t={x[2]:.4f}, CL={CL}")
    return np.append(x, [CL, J])

global path_points
path_points = []

bounds = [(0, 0.095),
          (0.2, 0.9),
          (0.1, 0.4)] #bounds within which the NACA generator returns smooth airfoil shapes
# x = optimize(x = np.array([0.04, 0.4, 0.12]))
def eq_cons(x):
    # return x[0] - 1/(200 * x[1]) #m - 1/p 
    return (6*(x[1] - 0.45))**2 + (12*(x[2]-0.2))**2 - 1

# solution = minimize(obj_function, 
#              [0.04, 0.6, 0.2], 
#              method='Nelder-Mead', 
#              bounds=bounds) 

constr = {'type': 'eq', 'fun': eq_cons}
solution = minimize(obj_function, 
             [0.04, 0.4 , 0.2], 
             method='trust-constr', 
             bounds=bounds, 
             constraints=constr,
             options={'maxiter':50}) 

naca = f"{int(solution.x[0]*100)}{int(solution.x[1]*10)}{int(solution.x[2]*100)}"
print(f"The most similar airfoil (not a valid solution!) \nNACA {naca}")
upper, lower = naca4digit_gen(solution.x[0], solution.x[1], solution.x[2])
fig1 = plt.figure(figsize=(6,2))
fig1.suptitle(f'Optimal solution: NACA {naca}')
ax = fig1.add_subplot()
ax.plot(upper[:,0], upper[:,1])
ax.plot(lower[:,0], lower[:,1])
ax.set_aspect('equal')

solution_space(resolution = 10, n_pts = 80, build = False, path_to_sol = path_points)
