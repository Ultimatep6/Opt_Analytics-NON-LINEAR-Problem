import matplotlib.pyplot as plt
import numpy as np
from alive_progress import alive_bar 
import time

import multiprocessing as mp
from multiprocessing.sharedctypes import Array
import ctypes

#flight params
V_inf = 15
alpha = np.deg2rad(20)
rho = 1.225

#sim params
n = 100 #number of panel vertices

def naca4digit_gen(M, P, T, n=100): #http://airfoiltools.com/airfoil/naca4digit
    beta = np.linspace(0, np.pi, n)
    x = (1-np.cos(beta))/2

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

def generate_sheet_points(airfoil, n=100):
    points_x = np.linspace(0, 1, n)
    interpolated = np.interp(points_x, airfoil[:,0], airfoil[:,1])
    return np.vstack([points_x, interpolated]).T

class Panel():
    def __init__(self, coord1, coord2):
        self.coord1 = coord1
        self.coord2 = coord2 
        self.midpoint = (coord1 + coord2)/2
        dx = (coord2[0] - coord1[0])
        dy = (coord2[1] - coord1[1])
        self.mag = np.sqrt(dx**2 + dy**2)

        self.beta = np.atan(dy/dx) - alpha
        self.lambd = 0
        self.tangent = np.array([dx, dy])/self.mag
        self.norm = np.array([-dy, dx])/self.mag
        # print(np.rad2deg(beta))

def generate_panels(points):
    gen_panels = []
    for point, next_point in zip(points, points[1:,:]): #exclude the last one
        gen_panels.append(Panel(point, next_point))

    return gen_panels

def panel_gammas(panels, V):
    N = len(panels)
    I = np.zeros((N,N)) #build the influence matrix to solve the system for vortex strengths
    b_vector = np.zeros((N,1))

    for i in range(len(panels)):
        for j in range(len(panels)):
            if i != j:
                r_ij = panels[i].midpoint - panels[j].midpoint
                r_ij_perp = np.array([-r_ij[1], r_ij[0]])

                I[i,j] = 1/(2*np.pi) * np.dot(r_ij_perp, panels[i].norm)/(np.linalg.norm(r_ij) ** 2)
            if i == j:
                I[i,j] = -0.5 #vortex influence at the control point itself will be equal to lambda/2

        b_vector[i] = -V * (np.cos(alpha)*panels[i].norm[0] +
                    np.sin(alpha)*panels[i].norm[1])

    #Kutta condition, at the edge vortex
    I[-1,:] = 0
    I[-1, 0] = 1
    I[-1,-1] = 1
    b_vector[-1] = 0
    # print(I.shape)
    
    gammas = np.linalg.solve(I, b_vector)
    circulation = sum([gammas[i] * panels[i].mag for i in range(len(gammas))])
    # print(circulation)

    C_l = 2/(V) * circulation

    #tangential velocities
    V = np.zeros(N)
    C_p = np.zeros(N)
    for i in range(len(panels)):
        ti = panels[i].tangent
        V_i = V_inf * (np.cos(alpha) * ti[0] + np.sin(alpha) * ti[1]) #freestream contribution to tangential velocity
        for j in range(len(panels)):
            if i != j:
                r_ij = panels[i].midpoint - panels[j].midpoint
                r_ij_perp = np.array([-r_ij[1], r_ij[0]])

                V_i += gammas[i]/(2*np.pi) * np.dot(r_ij_perp, ti)/(np.linalg.norm(r_ij) ** 2)
                # print(V_contr.shape)
        V[i] = V_i[0] #idk why V_i is an array lol

    C_p = 1 - (V/V_inf)**2 
    return C_l[0], C_p, np.array([panel.midpoint for panel in panels])

def airfoil_plot(airfoil, points, C_p, midpoints):
    """Airfoil shape plotter""" 
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(5,5))
    ax1.annotate("", xy=(0, 0), xytext=(-0.1*np.cos(alpha), -0.1*np.sin(alpha)), arrowprops=dict(fc="red", ec="red", width=1, headwidth=7.5))
    ax1.plot(airfoil[:,0], airfoil[:,1])
    ax1.plot(points[:,0], points[:,1], '-o')
    ax1.set_aspect('equal')

    # print(C_p)
    ax2.plot(midpoints[:,0], C_p)
    ax2.grid(True, which='both')
    ax2.axhline(y=0, color='k')

def main_method(max_camber, max_camber_pos, thickness, n_points):
    airfoil_top, airfoil_bottom = naca4digit_gen(max_camber, max_camber_pos, thickness)
    panels = []
    
    points_top = generate_sheet_points(airfoil_top, n=n_points)
    panels += generate_panels(points_top)[::-1] #we reverse the top vortices so that the first vortex is located at the trailing edge. Then, the Kutta condition is enforced at the trailing edge

    points_bottom = generate_sheet_points(airfoil_bottom, n=n_points)
    panels += generate_panels(points_bottom)

    C_l, C_p_list, midpoints = panel_gammas(panels, V_inf)
    # print(f'Lift Coeff per meter span: {C_l}')
    return C_l, C_p_list

def obj_function(x):
    # print(x)
    # target_C_l = 1
    max_camber, max_camber_pos, thickness = vars[x,:]
    #max camber %, max camber position %, thickness %
    tic = time.time()
    C_l, C_p_list = main_method(max_camber, max_camber_pos, thickness, n_points)
    toc = time.time()
    print(f'{vars[x,:]}: {(toc-tic):.2f}s')
    # obj = (C_l - target_C_l)**2 + 0.1*np.sum((C_p_list + 1)**2)
    output_shm[x] = C_l
    return C_l

def to_numpyarray(mp_arr):
    return np.frombuffer(mp_arr.get_obj(), dtype=np.float32)

# #start param
# x = np.array([0.02, 0.4, 0.12])
# epsilon = 1e-4
# eta = 0.01

# for k in range(40):
#     J, CL = obj_function(x, 1)
#     grad = np.zeros(3)

#     for i in range(3):
#         x_pert = x.copy()
#         x_pert[i] += epsilon
#         Jp,_ = obj_function(x_pert, 1)
#         grad[i] = (Jp - J)/epsilon

#     x = x - eta*grad
#     print(f"Iter {k}: J={J}, Jp={Jp} m={x[0]:.4f}, p={x[1]:.4f}, t={x[2]:.4f}, CL={CL}")
def main():
    global n_points
    resolution = 15
    n_points = 100

    m_vals = np.linspace(0.02, 0.1, resolution)
    p_vals = np.linspace(0.1, 0.6, resolution)
    t_vals = np.linspace(0.02, 0.40, resolution)
    n = resolution**3

    global vars
    m,p,t = np.meshgrid(m_vals, p_vals, t_vals, indexing='ij')
    vars = np.column_stack((m.ravel(), p.ravel(), t.ravel()))
    print(vars.shape)

    global output_shm
    output_shm = Array(ctypes.c_float, n, lock=True)
    iterator = range(int(n))

    NUMCORES = int(mp.cpu_count())
    avg = 0.9
    ops_per_core = n/NUMCORES
    print(f"Each core will process approx {ops_per_core} airfoils")
    print(f"Each airfoil process takes approx {avg}s to complete. Expect a total of {ops_per_core*avg} seconds.")

    tick = time.time()
    with mp.Pool(processes=NUMCORES, 
                initargs=[output_shm, n]) as p:
        p.map(obj_function, iterator)
    tock = time.time()

    results = to_numpyarray(output_shm)
    res = np.column_stack([vars, results])
    print(res.shape)
    print(f"Total time elapsed:{(tock-tick):.2f} seconds")

    np.savetxt(f"result_data_res{resolution}.csv", res, header='m,p,t,C_l', delimiter=',')
    return res

res = main()

# res = np.loadtxt(f'result_data_res{10}.csv', delimiter=',', skiprows=1)
# print(res.shape)

X = res[:,1]
Y = res[:,2]
Z = res[:,3]

ax = plt.figure().add_subplot(projection='3d')
ax.plot_trisurf(X, Y, Z)
ax.set_zlabel('C_l')
ax.set_zlim(-0.05,0.05)

plt.show()







