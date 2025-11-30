import matplotlib.pyplot as plt
import numpy as np

V = 5
S = 10

resolution = 25

alpha = np.deg2rad(15)
rho = 1.225
c_L = 2*np.pi * alpha #thin airfoil theory

c_L, rho = np.meshgrid(c_L, rho)
L = c_L * 0.5 * rho * V**2 * S

airfoil = np.loadtxt('user-000.csv', skiprows=1, delimiter=',')
maximum = np.max(airfoil[:,0])
# airfoil /= maximum
print(airfoil.shape)

airfoil_top, airfoil_bottom = np.array_split(airfoil, 2)
airfoil_top = airfoil_top[::-1,:]

def generate_points(airfoil, n=100):
    points_x = np.linspace(0, maximum, n)
    # points_x = 40
    # print(airfoil[:,0])
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

    # def delta_phi():

def generate_panels(points):
    gen_panels = []
    for point, next_point in zip(points, points[1:,:]): #exclude the last one
        gen_panels.append(Panel(point, next_point))

    return gen_panels

def panel_gammas(panels, V):
    N = len(panels)
    I = np.zeros((N,N)) #build theinfluence matrix to solve the system for vortex strength
    b_vector = np.zeros((N,1))

    for i in range(len(panels)):
        for j in range(len(panels)):
            if i != j:
                r_ij = panels[i].midpoint - panels[j].midpoint
                r_ij_perp = np.array([-r_ij[1], r_ij[0]])

                I[i,j] = 1/(2*np.pi) * np.dot(r_ij_perp, panels[i].norm)/(np.linalg.norm(r_ij) ** 2)
            if i == j:
                I[i,j] = -0.5 #vortex strength at the point itself will be equal to lambda/2

        b_vector[i] = -V * np.sin(panels[i].beta) 

    #Kutta condition, at the edge vortex
    I[-1,:] = 0
    I[-1, 0] = 1
    I[-1,-1] = 1
    b_vector[-1] = 0
    print(I)
    
    gammas = np.linalg.solve(I, b_vector)
    # print(gammas)

    circulation = sum([gammas[i] * panels[i].mag for i in range(len(gammas))])
    # print(circulation)

    C_l = 2/(V*maximum) * circulation
    print(f'Body Lift: {C_l}')

    return #gamma

def airfoil_plot(airfoil, points):
    """Airfoil shape plotter""" 
    plt.quiver(-15,-3, V*np.cos(alpha), V*np.sin(alpha), color='r')

    plt.plot(airfoil[:,0], airfoil[:,1])
    plt.plot(points[:,0], points[:,1], '-o')
    plt.gca().set_aspect('equal')

panels = []

n = 30

points_top = generate_points(airfoil_top, n=n)
panels += generate_panels(points_top)[::-1] #we reverse the top vortices so that the first vortex is located at the trailing edge. Then, the Kutta condition is enforced at the trailing edge

points_bottom = generate_points(airfoil_bottom, n=n)
panels += generate_panels(points_bottom)

# print(panels[0])
panel_gammas(panels, V)

airfoil_plot(airfoil_top, points_top)
airfoil_plot(airfoil_bottom, points_bottom)
plt.show()







