# Animation class for neural nets

from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np

#################
# Animation class
#################
class lcaAnimation():
    def __init__(self):
        self.fig = None
        self.ax = None 
        self.image_grid = None
        self.text0 = None
        self.n_images = None
    def setDataShape(self,data_shape):
        self.data_shape = data_shape
    def setColorMap(self,colormap):
        self.cmap = colormap
    def setMinMax(self,vmin,vmax):
        self.vmin,self.vmax = vmin,vmax
    def setDisplaySize(self,display_size):
        self.display_size = display_size
    def initFig(self):
        # Initialize visualization
        self.fig, self.ax = plt.subplots(self.display_size[0],self.display_size[1],figsize=(3,3))
        self.image_grid = self.ax.copy()
        self.text0 = plt.text(-170,-170,"")
        for b in range(self.display_size[0]):
             for c in range(self.display_size[1]):
                 self.image_grid[b,c] = self.ax[b,c].imshow(np.squeeze(np.zeros(self.data_shape)),cmap=self.cmap,vmin=self.vmin,vmax=self.vmax,interpolation="nearest")
                 self.ax[b,c].set_axis_off()
    def animate(self,arr,show=True,save=False,fname="",normalize=False):
        normalized_arr = arr
        if normalize:
            normalized_arr = arr - np.amin(arr,axis=(0,1,2),keepdims=True)
            normalized_arr /= np.amax(normalized_arr,axis=(0,1,2),keepdims=True)
        for b in range(self.display_size[0]):
             for c in range(self.display_size[1]):
                 scaled_image = np.squeeze(normalized_arr[self.display_size[0]*b+c]) 
                 self.image_grid[b,c].set_data(scaled_image)

        self.text0.set_text("")
        if show:
            self.fig.draw()
            self.fig.pause(0.01)
        if save: 
            self.fig.savefig(fname+".png")

class videoAnimation():
    def __init__(self):
        self.fig = None
        self.ax = None 
        self.image_grid = None
        self.text0 = None
        self.n_images = None
        self.data = None
    def setDataShape(self,data_shape):
        self.data_shape = data_shape
    def setColorMap(self,colormap):
        self.cmap = colormap
    def setMinMax(self,vmin,vmax):
        self.vmin = vmin
        self.vmax = vmax
    def setFps(self,fps):
        self.fps = fps
    def setDisplaySize(self,display_size):
        self.display_size = display_size
    def initFig(self):
        self.fig = plt.figure()
        self.ax = [[None for i in range(self.display_size[0])] for j in range(self.display_size[1])]
        for i in range(self.display_size[0]):
            for j in range(self.display_size[1]):
                self.ax[i][j] = self.fig.add_subplot(self.display_size[0],self.display_size[1],self.display_size[0]*i+j+1)
        self.image_grid = [self.ax[i][j].imshow(np.zeros(self.data_shape[1:])) for i in range(self.display_size[0]) for j in range(self.display_size[1])] 
    def initAnimation(self):
        for b in range(self.display_size[0]):
             for c in range(self.display_size[1]):
                 self.image_grid[self.display_size[0]*b+c] = self.ax[b][c].imshow(np.squeeze(np.zeros(self.data_shape[1:])),cmap=self.cmap,vmin=self.vmin,vmax=self.vmax,interpolation="nearest")
                 self.ax[b][c].set_axis_off()
        return self.image_grid
    def animate(self,i):
        for b in range(self.display_size[0]):
             for c in range(self.display_size[1]):
                 scaled_image = np.squeeze(self.data[self.display_size[0]*b+c,i]) 
                 self.image_grid[self.display_size[0]*b+c].set_array(scaled_image)
        return self.image_grid
    def makeAnimation(self,data,show=False,save=True,fname="",normalize=False):
        self.data = data
        if normalize:
            self.data = data - np.amin(data,axis=(1,2,3,4),keepdims=True)
            self.data /= np.amax(self.data,axis=(1,2,3,4),keepdims=True)
        anim = animation.FuncAnimation(self.fig,self.animate,init_func=self.initAnimation,frames=np.shape(data)[1],interval=int(1000./self.fps),blit=True)
        if save:
            anim.save(fname+".mp4")
        if show:
            self.plt.show()
