import matplotlib.pyplot as plt
import numpy as np

#flight params
V = 15
alpha = np.deg2rad(20)
rho = 1.225

#sim params
n = 10 #number of panel vertices

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

def generate_points(airfoil, n=100):
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
    print(I.shape)
    
    gammas = np.linalg.solve(I, b_vector)
    circulation = sum([gammas[i] * panels[i].mag for i in range(len(gammas))])
    # print(circulation)

    C_l = 2/(V) * circulation
    
    return C_l

def airfoil_plot(airfoil, points):
    """Airfoil shape plotter""" 
    fig, ax = plt.subplots(figsize=(5,5), layout='constrained')
    # ax.quiver(-0.2, -0.05, V*np.cos(alpha), V*np.sin(alpha), color='r')

    # ax.annotate(
    # "",
    # xy=(theta, r),
    # xytext=(theta, r + 0.5), # < arrow's length
    # arrowprops=dict(fc="C0", width=10, headwidth=25),
    # )
    ax.plot(airfoil[:,0], airfoil[:,1])
    ax.plot(points[:,0], points[:,1], '-o')
    ax.set_aspect('equal')

panels = []
#max camber %, max camber position %, thickness %
airfoil_top, airfoil_bottom = naca4digit_gen(0.06, 0.4, 0.08)
# print(airfoil_top.shape)
# maximum = 1

points_top = generate_points(airfoil_top, n=n)
panels += generate_panels(points_top)[::-1] #we reverse the top vortices so that the first vortex is located at the trailing edge. Then, the Kutta condition is enforced at the trailing edge

points_bottom = generate_points(airfoil_bottom, n=n)
panels += generate_panels(points_bottom)

C_l = panel_gammas(panels, V)
print(f'Lift Coeff per meter span: {C_l}')

airfoil_plot(np.concatenate([airfoil_top, airfoil_bottom[::-1]]), 
             np.concatenate([points_top, points_bottom[::-1]]))
plt.show()







