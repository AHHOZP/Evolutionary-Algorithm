import numpy as np
import random, math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Global Constant
time = 0
dt = 0.0005
gravity = np.array([0,0,-9.81])
omega = 1
kk,aa,bb,cc = 10000,2,0.2,math.pi



class Mass():
    def __init__(self,mass,position,vector=[0,0,0],accel=[0,0,0]):
        self.m = mass
        self.p = np.array(position)
        self.v = np.array(vector)
        self.a = np.array(accel)
        self.mus = 0.55
        self.muk = 0.4

class String():
    def __init__(self,k,a,b,c,l,m1,m2):
        self.k = k
        self.l0 = l
        self.l = self.l0
        self.m1 = m1
        self.m2 = m2
        self.a = a
        self.b = b
        self.c = c

    def breath(self):
        self.l = self.a * self.l0 + self.b * self.l0 * math.sin(omega * time * math.pi * 20 + self.c)

class Cube():
    def __init__(self,vertex,edge):
        self.vertex = vertex
        self.edge = edge
        self.size_vertex = range(len(self.vertex))
        self.size_edge = range(len(self.edge))
        self.center0_xy = self.cal_center_xy()

    def cal_center_xy(self):
        size = len(self.vertex)
        x = np.sum([v0.p[0] for v0 in self.vertex])/size
        y = np.sum([v1.p[1] for v1 in self.vertex])/size
        return np.array([x,y])

    def evaluate(self):
        for i in range(10000):
            self.force_clear()
            self.force_cal()
            self.vector_cal()
            self.position_cal()
        self.fitness = np.linalg.norm(self.center0_xy - self.cal_center_xy())

    
    def con_st(self, string):
        return self.vertex[string.m1], self.vertex[string.m2]
        
    def apply_force(self,F,m):
        self.vertex[m].a = np.add(self.vertex[m].a, np.divide(F,self.vertex[m].m))

    def apply_gravity(self,m):
        self.vertex[m].a = np.add(self.vertex[m].a, gravity)

    def apply_vector(self,m):
        #self.vertex[m].v = np.add(self.vertex[m].v, np.multiply(self.vertex[m].a, dt))
        # dampending
        self.vertex[m].v = np.multiply(np.add(self.vertex[m].v, np.multiply(self.vertex[m].a, dt)), 0.995)

    def apply_posi(self,m):
        self.vertex[m].p = np.add(self.vertex[m].p, np.multiply(self.vertex[m].v, dt))
    
    def apply_elastic(self):
        for i in self.size_edge:
            self.edge[i].breath()
            a,b = self.con_st(self.edge[i])
            m21 = np.subtract(a.p, b.p)
            dist = np.linalg.norm(m21)
            F = self.edge[i].k * (self.edge[i].l - dist)
            F21 = np.multiply(m21, F/dist)
            self.apply_force(F21,self.edge[i].m1)
            self.apply_force(-F21,self.edge[i].m2)

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

class GA_Rob():
    def __init__(self):
        self.population = []
        self.generation = 1
        self.enviroment = 0.0
        self.size_population = 50
        
        self.crossRate = 0.8
        self.mutationRate = 0.01
        
    def create_individual(self):

        a,b,c = 0,0,0.5773
        cube_vertex = [Mass(1.5, [a+0, b+0, c+0]),
                    Mass(1, [a+0, b+0, c+1.1547]),
                    Mass(1, [a+1.1547, b+0, c-0.5773]),
                    Mass(1, [a-0.5773, b-0.5, c-0.5773]),
                    Mass(1, [a-0.5773, b+0.5, c-0.5773])]

        cube_edge = []
        for i in range(5):
            for j in range(4-i):
                ij = cube_vertex[i].p - cube_vertex[j+i+1].p
                kkk,aaa,bbb,ccc = 0,0,0,0
                while kkk < 100:
                    kkk = kk*np.random.normal(loc=0.5,scale=0.1) + 1000
                while aaa <= bbb:
                    while aaa <= 0:
                        aaa = aa*np.random.normal(loc=1,scale=0.1)
                    bbb = bb*np.random.normal(loc=0.5,scale=0.1)
                ccc = cc*np.random.normal(scale=0.1)
                cube_edge.append(String(kkk,aaa,bbb,ccc,np.linalg.norm(ij),i,(j+i+1)))

        cube = Cube(cube_vertex,cube_edge)

        return cube

    def first_gen(self):
        for i in range(self.size_population):
            individual = self.create_individual()
            individual.evaluate()
            self.population.append(individual)

    def cal_env(self):
        self.enviroment = 0.0
        self.population.sort(key=lambda x: x.fitness)
        self.best = self.population[0]
        for i in range(len(self.population)):
            self.enviroment += self.population[i].fitness

    def selection_T(self):
        # Selection pressure
        sp = 15
        prob = 0.15
        pp = random.sample(self.population,sp)
        pp.sort(key=lambda x: x.fitness)
        if random.random() > prob:
            return pp[-1]
        else :
            if random.random() > prob:
                return pp[-2]
            else:
                return pp[-3] 


    def selection_RW(self):
        self.cal_env()
        new_population = []
        pp = []
        n=5
        for i in range(len(self.population)/n):
            pp.append(random.random())
        pp.sort()
        sp = 0
        num = 0
        for p in pp:
            while(p>sp):
                sp += self.population[num].fitness/self.enviroment
                num += 1
            new_population.append(self.population[num-1])
        return new_population

    def cross(self, ind1, ind2):
        prob_parent = ind1.fitness/(ind1.fitness + ind2.fitness)
        new_ind1 = ind1
        new_ind2 = ind2
        for i in ind1.size_edge:
            if random.random() < prob_parent:
                new_ind2.edge[i] = ind1.edge[i]
            else:
                new_ind1.edge[i] = ind2.edge[i]
        return new_ind1, new_ind2

    def mutation(self, ind):
        for i in ind.size_edge:
            if random.random() < self.mutationRate:
                ind.edge[i].k += np.random.normal(scale=0.1) * kk
                if ind.edge[i].k < 100:
                    ind.edge[i].k = 100
            if random.random() < self.mutationRate:
                ind.edge[i].a += np.random.normal(scale=0.1) * aa
            if random.random() < self.mutationRate:
                ind.edge[i].b += np.random.normal(scale=0.1) * bb
            if random.random() < self.mutationRate:
                ind.edge[i].c += np.random.normal(scale=0.1) * cc
            while ind.edge[i].a <= ind.edge[i].b:
                while ind.edge[i].a <= 0:
                    ind.edge[i].a = aa*np.random.normal(loc=1,scale=0.1)
                ind.edge[i].b = bb*np.random.normal(loc=0.2,scale=0.1)
        return ind
                    

    def generate_next(self):
        new_population = []
        while len(new_population) < self.size_population:
            parent1 = self.selection_T()
            if random.random() < self.crossRate:
                parent2 = self.selection_T()
                child1, child2 = self.cross(parent1,parent2)
                child1 = self.mutation(child1)
                child1.evaluate()
                child2 = self.mutation(child2)
                child2.evaluate()
                new_population.append(child1)
                new_population.append(child2)
            else:
                child1 = self.mutation(parent1)
                child1.evaluate()
                new_population.append(child1)
            new_population.sort(reverse = True, key=lambda x: x.fitness)
            new_population = new_population[:self.size_population]
        self.population = new_population
        self.generation += 1
        
    def envolve(self):
        self.first_gen()
        while self.generation < 5:
            self.generate_next()
        return self.population

GARobot = GA_Rob()

last_generation = GARobot.envolve()

last_generation.sort(reverse = True, key=lambda x: x.fitness)

f = open('../best.txt','w')
f.write('best robot parameters\n')

for i in range(5):
    f.write(str(i)+str("th indi:")+'\n')
    for j in range(len(last_generation[i].edge)):
        sk = last_generation[i].edge[j].k
        sa = last_generation[i].edge[j].a
        sb = last_generation[i].edge[j].b
        sc = last_generation[i].edge[j].c
        f.write(str(j+1)+str("th string:")+'\n')
        f.write(str("k=")+str(sk)+'\n')
        f.write(str("a=")+str(sa)+'\n')
        f.write(str("b=")+str(sb)+'\n')
        f.write(str("c=")+str(sc)+'\n')

f.close()