from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from matplotlib.widgets import Button
from pymss import EnvManager



use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class Graph(object):
    def __init__(self, env, dim, path):
        # self.ind = 0
        self.dim = dim
        self.count = 0
        self.n = 100 # resolution

        if dim == 1:
            fig = plt.figure()
            self.ax = fig.add_subplot(1,1,1)
        else:
            fig = plt.figure()
            self.ax = fig.gca(projection='3d')
            self.ax.zaxis.set_major_locator(LinearLocator(10))
            self.ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))


        # Domain Range 
        self.lo = env.GetMinV()
        self.hi = env.GetMaxV()


        self.X = []
        for i in range(dim):
            self.X.append(np.arange(self.lo[i], self.hi[i],(self.hi[i] - self.lo[i])/self.n))
            
        if dim == 2:
            self.X[0], self.X[1] = np.meshgrid(self.X[0], self.X[1])
        else:
            self.X = self.X[0]   
    
        self.marginal_model = MarginalNN(dim)
        if use_cuda:
            self.marginal_model.cuda()
        

        if dim == 1:
            self.V = np.zeros(self.n)
        elif dim == 2:
            self.V = np.zeros((self.n, self.n))
        

        self.loadandplot(path)

    def next(self, event):
        self.count += 1
        # plt.clf()
        ax.clear()
        self.loadandplot(path)

    def prev(self, event):
        self.count -= 1
        if self.count < 0:
            self.count = 0
        self.surf.remove()
        self.loadandplot()

    def loadandplot(self, path):
        self.marginal_model.load(path)

        if self.dim == 2:
            for i in range(self.n):
                for j in range(self.n):
                    state = FloatTensor(np.array([self.X[0][i][j], self.X[1][i][j]]))
                    if self.X[0][i][j] <= self.X[1][i][j]:
                        self.V[i][j] = self.marginal_model(state).cpu().detach().numpy()
                    else:
                        self.V[i][j] = 0
            self.surf = self.ax.plot_surface(self.X[0], self.X[1], self.V, linewidth = 0, antialiased = False)
        else:
            for i in range(self.n):
                state = FloatTensor(np.array([self.X[i]]))
                self.V[i] = self.marginal_model(state).cpu().detach().numpy()
            self.ax.plot(self.X, self.V)

        
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--marginal_model', help='model path')
    parser.add_argument('-d', '--meta', help = 'meta file')
	
    args = parser.parse_args()

    if args.meta == None or args.marginal_model == None:
        print('Wrong Arguments')
        exit()

    path = args.marginal_model
    env = EnvManager(args.meta, 1)
    dim = env.GetNumParamState()

    callback = Graph(env, dim, path)

    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])

    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next)
    
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev)

    plt.show()
    
