# Author: Swadhin Agrawal
# E-mail: swadhin12a@gmail.com

import numpy as np
import copy

class VoterModel:
    def __init__(self):
	# def __init__(self,animator,robots,options,p):
		# self.animator = animator
        self.consensus = False
        self.best_option = None
        self.iterations = 0
		
    def dissemination(self,robots,options,p):
        t = 0
        yes_respondents = []
        for r in robots:
            if r.response==1:
                yes_respondents.append(r)

        while self.consensus == False:
			# Fails when total number of yes reponding agents is below neighbours number
			# Number of neighbours considered affects the consensus achievment time and success
            consensus_limit = len(yes_respondents)
            if consensus_limit < 1:
                break
            talker = np.random.choice(range(len(yes_respondents)),1,replace = False)[0]

            listener = np.random.choice(range(len(robots)),1,replace = False)[0]

            if yes_respondents[talker].id != robots[listener].id:
                if robots[yes_respondents[talker].id].opt != robots[listener].opt:

                    robots[listener].opt = copy.copy(robots[yes_respondents[talker].id].opt)
					
                    robots[listener].response = 1

					# robots[listener].patch.set_color(options[robots[listener].opt-1].color)
                    if robots[listener] not in yes_respondents:
                        yes_respondents.append(robots[listener])

					# plt.pause(0.001)
					# plt.show()

                same = robots[yes_respondents[talker].id].opt
                counter = 1
                opt_counter1 = []
				
                for r in yes_respondents:
                    opt_counter1.append(r.opt-1)
                    if same == r.opt:
                        counter += 1
						
                opt_counter = 100*np.bincount(opt_counter1)/len(yes_respondents)

				# for i in range(len(opt_counter)):
				# 	if isinstance(options[i].progress_bar,type(None)) != True:
				# 		options[i].progress_bar.remove()
				# 	options[i].progress_bar = self.animator.ax.bar(50 + i*10,opt_counter[i],color = options[i].color,alpha = 0.3,width = 5)			
				
                if counter/(p.num_robots) >= 0.99:
                    self.consensus = True
                    self.best_option = same - 1

                t += 1
        self.iterations = t