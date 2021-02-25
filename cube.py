from numba import cuda
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from IPython.display import HTML

#global time, dt, gravity
time = 0
dt = 0.0005
gravity = np.array([0,0,-9.81])

class Mass():
    def __init__(self,mass,position,vector=[0,0,0],accel=[0,0,0]):
        self.m = mass
        self.p = np.array(position)
        self.v = np.array(vector)
        self.a = np.array(accel)
        self.mus = 0.5
        self.muk = 0.4

class String():
    def __init__(self,k,l,m1,m2):
        self.k = k
        self.l = l
        self.m1 = m1
        self.m2 = m2

class Cube():
    def __init__(self,vertex,edge):
        self.vertex = vertex
        self.edge = edge
        self.size_vertex = range(len(self.vertex))
        self.Ek = self.energy_k()
        self.Ep = self.energy_g() +self.energy_e()
        self.Et = self.Ek + self.Ep
    
    def con_st(self, string):
        return self.vertex[string.m1], self.vertex[string.m2]
        
    def apply_force(self,F,m):
        self.vertex[m].a = np.add(self.vertex[m].a, np.divide(F,self.vertex[m].m))

    def apply_gravity(self,m):
        self.vertex[m].a = np.add(self.vertex[m].a, gravity)

    def apply_vector(self,m):
        #self.vertex[m].v = np.add(self.vertex[m].v, np.multiply(self.vertex[m].a, dt))
        # dampending
        self.vertex[m].v = np.multiply(np.add(self.vertex[m].v, np.multiply(self.vertex[m].a, dt)), 0.999)

    def apply_posi(self,m):
        self.vertex[m].p = np.add(self.vertex[m].p, np.multiply(self.vertex[m].v, dt))
    
    def apply_elastic(self):
        for string in self.edge:
            a,b = self.con_st(string)
            m21 = np.subtract(a.p, b.p)
            dist = np.linalg.norm(m21)
            F = string.k * (string.l - dist)
            F21 = np.multiply(m21, F/dist)
            self.apply_force(F21,string.m1)
            self.apply_force(-F21,string.m2)

    def force_cal(self):
        self.apply_elastic()
        for i in self.size_vertex:
            self.apply_gravity(i)
            # apply friction
            if self.vertex[i].p[2]<=0:
                if self.vertex[i].a[2]<0:
                    aH = np.array([self.vertex[i].a[0], self.vertex[i].a[1],0])
                    aH_norm = np.linalg.norm(aH)
                    if aH_norm < - self.vertex[i].a[2] * self.vertex[i].mus:
                        self.vertex[i].a[0] = 0
                        self.vertex[i].a[1] = 0
                        self.vertex[i].v[0] = 0
                        self.vertex[i].v[1] = 0
                    else:
                        f = np.multiply(np.multiply(self.vertex[i].muk, self.vertex[i].a[2]),self.vertex[i].m)
                        f_dire = np.divide(aH, aH_norm)
                        friction = np.multiply(-f_dire, f)
                        self.apply_force(friction, i)
            # apply ground force
            if self.vertex[i].p[2]<0:
                Fc = np.array([0,0,-20000*self.vertex[i].p[2]])
                self.apply_force(Fc,i)

    def vector_cal(self):
        for i in self.size_vertex:
            self.apply_vector(i)

    def position_cal(self):
        for i in self.size_vertex:
            self.apply_posi(i)    

    def force_clear(self):
        for i in self.size_vertex:
            self.vertex[i].a = np.array([0,0,0])

    def energy_cal(self):
        self.Ek = self.energy_k()
        self.Ep = self.energy_g() +self.energy_e()
        self.Et = self.Ek + self.Ep        

    def energy_g(self):
        Eg = 0
        for mass in self.vertex:
            Eg += mass.m * (-gravity[2]) * mass.p[2]
        return Eg

    def energy_e(self):
        Ee = 0
        for string in self.edge:
            a,b = self.con_st(string)
            m21 = a.p-b.p
            dist = np.linalg.norm(m21)
            Ee += 0.5 * string.k * (string.l - dist)**2
        return Ee

    def energy_k(self):
        Ek = 0
        for mass in self.vertex:
            Ek += 0.5 * mass.m * np.linalg.norm(mass.v)**2
        return Ek

    def plot_energy(self):
        plt.plot(self.Ep, t, 'b')
        plt.plot(self.Ek, t, 'y')
        plt.plot(self.Et, t, 'r')
        plt.show()

    def plot_cube(self):
        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        ax.lines= []
        ax.collections= []
        point = []
        for i in range(3):
            point.append([x.p[i] for x in self.vertex])

        ax.scatter(point[0],point[1],point[2], c='r', marker='o')

        for string in self.edge:
            line = []
            a,b = self.con_st(string)
            for i in range(3):
                line.append([a.p[i],b.p[i]])
            ax.plot(line[0],line[1],line[2], color='b')

        plt.show()

'''def create_cube():
    cube_vertex = [Mass(0.1, [0, 0, 0]),
                Mass(0.1, [0, 0, 0.1]),
                Mass(0.1, [0, 0.1, 0]),
                Mass(0.1, [0.1, 0, 0]),
                Mass(0.1, [0.1, 0.1, 0]),
                Mass(0.1, [0.1, 0, 0.1]),
                Mass(0.1, [0, 0.1, 0.1]),
                Mass(0.1, [0.1, 0.1, 0.1])]

    cube_edge = []
    for i in range(8):
        for j in range(7-i):
            ij = cube_vertex[i].p - cube_vertex[j+i+1].p
            cube_edge.append(String(10000,np.linalg.norm(ij),i,(j+i+1)))

    cube = Cube(cube_vertex,cube_edge)

    return cube

cube = create_cube()'''

'''def create_bouncing_cube():
    a,b,c = 0,0,0.05
    cube_vertex = [Mass(0.1, [a+0, b+0, c+0]),
                Mass(0.1, [a+0, b+0, c+0.1]),
                Mass(0.1, [a+0, b+0.1, c+0]),
                Mass(0.1, [a+0.1, b+0, c+0]),
                Mass(0.1, [a+0.1, b+0.1, c+0]),
                Mass(0.1, [a+0.1, b+0, c+0.1]),
                Mass(0.1, [a+0, b+0.1, c+0.1]),
                Mass(0.1, [a+0.1, b+0.1, c+0.1])]

    cube_edge = []
    for i in range(8):
        for j in range(7-i):
            ij = cube_vertex[i].p - cube_vertex[j+i+1].p
            cube_edge.append(String(10000,np.linalg.norm(ij),i,(j+i+1)))

    cube = Cube(cube_vertex,cube_edge)

    return cube

cube = create_bouncing_cube()'''
def create_individual():

    a,b,c = 0,0,2.5773
    cube_vertex = [Mass(1.5, [a+0, b+0, c+0]),
                Mass(1, [a+0, b+0, c+1.1547]),
                Mass(1, [a+1.1547, b+0, c-0.5773]),
                Mass(1, [a-0.5773, b-0.5, c-0.5773]),
                Mass(1, [a-0.5773, b+0.5, c-0.5773])]

    cube_edge = []
    for i in range(5):
        for j in range(4-i):
            ij = cube_vertex[i].p - cube_vertex[j+i+1].p
            cube_edge.append(String(2000,np.linalg.norm(ij),i,(j+i+1)))

    cube = Cube(cube_vertex,cube_edge)

    return cube
cube = create_individual()



fig = plt.figure(figsize=(1904, 1071))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim3d([-0.5, 0.5])
ax.set_xlabel('X')

ax.set_ylim3d([-0.5, 0.5])
ax.set_ylabel('Y')

ax.set_zlim3d([0, 1.0])
ax.set_zlabel('Z')

ax.set_title('Bouncing Cube')

time_template = 'time = %.2fs'
time_text = ax.text(0, 0, 0.5, '')
time_text.set_text('')

aa=[]
bb=[]
cc=[]
tt=[]

# bouncing cube spin
'''cube.vertex[0].v = np.array([.3,-.3, .5])
cube.vertex[1].v = np.array([.3,-.3, .5])
cube.vertex[2].v = np.array([-.3,-.3,0])
cube.vertex[3].v = np.array([.3,.3,0])
cube.vertex[4].v = np.array([-.3,.3,0])
cube.vertex[5].v = np.array([.3,.3,0])
cube.vertex[6].v = np.array([-.3,-.3,0])
cube.vertex[7].v = np.array([-.3,.3,0])'''

def update_cube(frame):
    global time
    aa.append(cube.Ep)
    bb.append(cube.Ek)
    cc.append(cube.Et)
    tt.append(time)
    time_text.set_text(time_template % (time))
    
    cube.plot_cube()
    for i in range(100):
        cube.force_clear()
        cube.force_cal()
        cube.vector_cal()
        cube.position_cal()
        time = time + dt
        
    cube.energy_cal()



ani = animation.FuncAnimation(fig, update_cube, frames=10000, fargs=None,
                                   interval=50, blit=False)
plt.show()

def plot_energy(aa,bb,cc,tt):
    plt.plot(tt, aa, 'b', label="potential energy")
    plt.plot(tt, bb, 'r', label="kinetic energy")
    plt.plot(tt, cc, 'y', label="total energy")
    plt.xlabel('Time(s)')
    plt.ylabel('Energy(J)')
    plt.legend(bbox_to_anchor=(0.75, 1.18), loc='upper left')
    plt.show()
    
plot_energy(aa,bb,cc,tt)