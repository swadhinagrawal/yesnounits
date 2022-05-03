# Author: Swadhin Agrawal
# E-mail: swadhin12a@gmail.com

import numpy as np
import copy

class DecisionMaking:
	def __init__(self,robots,options,p):
	# def __init__(self,animator,robots,options,p):
		# self.animator = animator
		self.ref_best = None

		desired_responses = np.zeros_like(options)
		qualities = np.array([i.quality for i in options])
		votes = [40,39,10,5,2]
		for i in votes:
			maximums = np.where(qualities==max(qualities))[0]
			choosen = np.random.choice(maximums)
			desired_responses[choosen] += i
			qualities[choosen] = -100

		shuffled_r = copy.copy(robots)
		np.random.shuffle(shuffled_r)
		for r in shuffled_r:
			options[r.assigned_opt-1].assigned_count = 0
		for r in shuffled_r:
			if options[r.assigned_opt-1].assigned_count<=desired_responses[r.assigned_opt-1]:
				robots[r.id].response = 1
				options[r.assigned_opt-1].assigned_count += 1
				robots[r.id].opt = robots[r.id].response*robots[r.id].assigned_opt
			else:
				robots[r.id].response = 0
				robots[r.id].opt = robots[r.id].response*robots[r.id].assigned_opt
				options[r.assigned_opt-1].assigned_count += 1
		# 		r.patch.set_color(options[r.assigned_opt-1].color)
		# plt.pause(0.001)

	def compare_with_best(self,options,best_option):
		opts = []
		for o in range(len(options)):
			opts.append(options[o].quality)
		best_list = np.array(np.where(opts == max(opts)))[0]
		opt_choosen = np.random.randint(0,len(best_list))
		self.ref_best = best_list[opt_choosen]
		if self.ref_best==best_option:
			return 1
		else:
			return 0