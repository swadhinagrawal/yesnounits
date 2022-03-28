# Author: Swadhin Agrawal
# E-mail: swadhin12a@gmail.com

import numpy as np
import matplotlib.pyplot as plt

class Animator:
	def __init__(self,robots,options,p):
		self.fig,self.ax = plt.subplots()
		self.ax.set_aspect('equal')
		for o in options:
			c = plt.Circle((o.pose[0],o.pose[1]),2*o.quality/p.robot_size,fc = o.color,edgecolor = 'black',linewidth=1)
			self.ax.add_patch(c)
		for r in robots:
			r.patch = plt.Circle((r.pose[0],r.pose[1]), p.robot_size/10,fc = 'maroon',ec = 'maroon')
			self.ax.add_patch(r.patch)

		self.ax.set_xlim(-20,120)
		self.ax.set_ylim(-20,120)
		plt.show()