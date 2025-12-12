import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from alive_progress import alive_bar 
import time

import multiprocessing as mp
from multiprocessing.sharedctypes import Array
import ctypes

#flight params
V_inf = 15
alpha = np.deg2rad(15)
rho = 1.225

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

# def generate_sheet_points(airfoil, n=100):
#     points_x = np.linspace(0, 1, n)
#     interpolated = np.interp(points_x, airfoil[:,0], airfoil[:,1])
#     return np.vstack([points_x, interpolated]).T

def generate_sheet_points(airfoil, n=100):
    # sort by x to guarantee monotonic xp for np.interp
    idx = np.argsort(airfoil[:,0])
    xp = airfoil[idx, 0]
    yp = airfoil[idx, 1]

    # remove duplicates in xp
    unique_x, unique_idx = np.unique(xp, return_index=True)
    xp = unique_x
    yp = yp[unique_idx]

    # points_x = np.linspace(0.0, 1.0, n)
    beta = np.linspace(0, np.pi, n)
    points_x = (1-np.cos(beta))/2 #cosine spacing for better lead/trailing edge definition
    interpolated = np.interp(points_x, xp, yp)
    return np.vstack([points_x, interpolated]).T

class Panel():
    def __init__(self, coord1, coord2, top_bottom):
        self.coord1 = coord1
        self.coord2 = coord2 
        self.midpoint = (coord1 + coord2)/2
        dx = (coord2[0] - coord1[0])
        dy = (coord2[1] - coord1[1])
        self.mag = np.sqrt(dx**2 + dy**2)

        self.beta = np.atan2(dy, dx) - alpha 
        self.tangent = np.array([dx, dy])/self.mag 
        if top_bottom:
            self.norm = np.array([-dy, dx])/self.mag 
        else:
            self.norm = -np.array([-dy, dx])/self.mag 
        # print(np.rad2deg(beta))

def generate_panels(points, top_bottom):
    gen_panels = []
    for point, next_point in zip(points, points[1:,:]): #exclude the last one
        gen_panels.append(Panel(point, next_point, top_bottom=top_bottom))

    return gen_panels

def panel_gammas(panels, V_inf):
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
                I[i,j] = 0 #vortex influence at the control point itself will be none

        b_vector[i] = -V_inf * (np.cos(alpha)*panels[i].norm[0] +
                    np.sin(alpha)*panels[i].norm[1])

    #Kutta condition, at the edge vortex
    N_top = len(panels)//2
    I[-1,:] = 0
    I[-1, 0] = 1
    I[-1,N_top] = 1
    b_vector[-1] = 0
    # print(I.shape)
    
    gammas = np.linalg.solve(I, b_vector)
    circulation = sum([gammas[i] * panels[i].mag for i in range(len(gammas))])
    # print(circulation)

    C_l = -2/(V_inf) * circulation

    #tangential velocities
    V = np.zeros(N)
    C_p = np.zeros(N)
    for i in range(len(panels)):
        ti = panels[i].tangent
        V_i = V_inf * (np.cos(alpha) * ti[0] + np.sin(alpha) * ti[1]) + gammas[i]/2 #freestream contribution to tangential velocity plus itself
        for j in range(len(panels)):
            if i != j:
                r_ij = panels[i].midpoint - panels[j].midpoint
                r_ij_perp = np.array([-r_ij[1], r_ij[0]])

                V_i += gammas[j]/(2*np.pi) * np.dot(r_ij_perp, ti)/(np.linalg.norm(r_ij) ** 2)
                # print(V_contr.shape)
        V[i] = V_i[0] #idk why V_i is an array lol

    C_p = 1 - (V/V_inf)**2 
    return C_l[0], C_p, gammas

def airfoil_plot(fig, panels, x):
    """Airfoil shape plotter""" 
    # ax1.annotate("", xy=(0, 0), xytext=(-0.1*np.cos(alpha), -0.1*np.sin(alpha)), arrowprops=dict(fc="red", ec="red", width=1, headwidth=7.5))
    points = np.array([p.coord1 for p in panels])
    midpoints = np.array([p.midpoint for p in panels])
    norms = np.array([p.norm for p in panels])

    ax1 = fig.add_subplot(211)
    scatter = ax1.scatter(points[:,0], points[:,1], c = [i for i in range(points.shape[0])])
    fig.colorbar(scatter)
    ax1.quiver(midpoints[:,0], midpoints[:,1], norms[:,0], norms[:,1])
    # ax1.set_ylim(-0.5, 0.5)
    ax1.set_aspect('equal')

def cp_plot(fig, panel_top, panel_bottom, Cp, gammas):
    upper_midpoints = np.array([p.midpoint for p in panel_top])
    lower_midpoints = np.array([p.midpoint for p in panel_bottom])

    cu, cl = np.split(Cp, 2)
    gu, gl = np.split(gammas, 2)

    ax2 = fig.add_subplot(212)
    ax2.plot(upper_midpoints[:,0], cu, '-o', label='Cp upper', color = 'orange')
    ax2.plot(lower_midpoints[:,0], cl, '-o', label='Cp lower', color = 'blue')
    ax2.plot(upper_midpoints[:,0], gu, '-o', label='gammas upper', color = 'orange', alpha = 0.3)
    ax2.plot(lower_midpoints[:,0], gl, '-o', label='gammas lower', color = 'blue', alpha = 0.3)
    ax2.set_xlabel('x/c')
    ax2.set_ylabel('C_p')
    ax2.invert_yaxis()            # conventional: negative-upwards for Cp plots
    ax2.legend()
    ax2.grid(True)

def main_method(max_camber, max_camber_pos, thickness, n_points):
    tic = time.time()
    x = [max_camber, max_camber_pos, thickness]
    airfoil_top, airfoil_bottom = naca4digit_gen(max_camber, max_camber_pos, thickness)
    panels = []
    
    points_top = generate_sheet_points(airfoil_top, n=n_points)
    # print("top panels")
    panels_top = generate_panels(points_top, top_bottom=True)[::-1] #we reverse the top vortices so that the first vortex is located at the trailing edge. Then, the Kutta condition is enforced at the trailing edge
    panels += panels_top

    points_bottom = generate_sheet_points(airfoil_bottom, n=n_points)
    # print("bottom panels")
    panels_bottom = generate_panels(points_bottom, top_bottom=False)
    panels += panels_bottom


    C_l, C_p_list, gammas = panel_gammas(panels, V_inf)
    toc = time.time()
    # print(f'{x[0]:.3f} {x[1]:.3f} {x[2]:.3f}: {C_l:.4f} {(toc-tic):.2f}s')

    fig = plt.figure()
    fig.suptitle(f"max camber: {x[0]:.3f}, max camber pos: {x[1]:.3f}, thickness: {x[2]:.3f}")
    airfoil_plot(fig, panels, x)
    cp_plot(fig, panels_top, panels_bottom, C_p_list, gammas)
    print(f'Lift Coeff per meter span: {C_l}')    
    plt.show()
    return C_l, C_p_list

def obj_function(x):
    # print(x)
    # target_C_l = 1
    max_camber, max_camber_pos, thickness = vars[x,:]
    #max camber %, max camber position %, thickness %
    
    C_l, C_p_list = main_method(max_camber, max_camber_pos, thickness, n_points)
    # obj = (C_l - target_C_l)**2 + 0.1*np.sum((C_p_list + 1)**2)
    output_shm[x] = C_l
    return C_l

def to_numpyarray(mp_arr):
    return np.frombuffer(mp_arr.get_obj(), dtype=np.float32)

def solution_space(build, resolution, n_pts):
    if build:
        global n_points
        resolution = resolution
        n_points = n_pts

        m_vals = np.linspace(0, 0.095, resolution)
        p_vals = np.linspace(0, 0.9, resolution)
        t_vals = np.linspace(0.01, 0.40, resolution)
        n = resolution**3

        global vars
        m,p,t = np.meshgrid(m_vals, p_vals, t_vals, indexing='ij')
        vars = np.column_stack((m.ravel(), p.ravel(), t.ravel()))[::-1,:]
        print(vars.shape)

        global output_shm
        output_shm = Array(ctypes.c_float, n, lock=True)
        iterator = range(int(n))

        # NUMCORES = int(mp.cpu_count())
        NUMCORES = 8
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

        np.savetxt(f"results/maps/result_data_res{resolution}.csv", res, header='m,p,t,C_l', delimiter=',')
    
    else: 
        res = np.loadtxt(f'results/maps/result_data_res{resolution}.csv', delimiter=',')
        print(res.shape)

    # print(res[:,0])
    header = ['m', 'p', 't', 'C_l']
    X = res[:,0]
    Y = res[:,1]
    Z = res[:,2]
    vals = res[:,3]

    print(vals.shape)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # scat = ax.scatter(X, Y, Z, c=vals, cmap='pink')
    # fig.colorbar(scat, ax=ax)
    ax.plot_trisurf(X,Y,vals)

    ax.set_xlabel(header[0])
    ax.set_ylabel(header[2])
    ax.set_zlabel(header[3])
    ax.set_zlim(-1,5)
    plt.show()

# solution_space(resolution = 10, n_pts = 80, build = True)
main_method(0.06, 0.4, 0.12, 20)







