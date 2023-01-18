#This script uses pymoo and NSGA2 to optimise our favourite system

from pymoo.optimize import minimize
from SA import *
import numpy as np
import pickle

#Problem 
#Algorithm for minimisation
#Stop criteria

from pymoo.core.problem import Problem

#read files
#Load PDF for latency
kdefile = open('kdePickle','rb')
lat_kde = pickle.load(kdefile)
kdefile.close()

lanfile = open('LAN_Pickle','rb')
lat_lan = pickle.load(lanfile)
lanfile.close()

pbadFile = open('pbadPickle','rb')
p_bad = pickle.load(pbadFile)
pbadFile.close()

params = (lat_kde,lat_lan,p_bad)

class ProblemWrapper(Problem):
    def _evaluate(self, designs, out, *args, **kwargs):

        res = []
        for design in designs:
            res.append(obj_wrapper(design, *params))
        out['F'] = np.array(res)


#CONSTRAINTS
bcrit = 3
wp = 200
buf_min = bcrit+1
buf_max = wp
fact_min = 0
fact_max = 1

#GA Settings
p_size = 200
g_size = 50

problem = ProblemWrapper(n_var=3, n_obj=4, xl=[buf_min,fact_min,fact_min], xu=[buf_max,fact_max,fact_max])

#Algorithm
from pymoo.algorithms.moo.nsga2 import NSGA2

#do MILP things here
algorithm = NSGA2(pop_size=p_size)

stop_criteria = ('n_gen',g_size)

results = minimize(
    problem=problem,
    algorithm=algorithm,
    termination=stop_criteria
)
print(" POP: "+str(p_size)+"   "+str(g_size)+" Generations")
print("         FINISHED           ")
print(results.F)
print("             INPUTS        ")
print(results.X)