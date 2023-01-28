#This script uses pymoo and NSGA2 to optimise our favourite system

from pymoo.optimize import minimize
from SA import *
import numpy as np
import pickle

#Problem 
#Algorithm for minimisation
#Stop criteria

from pymoo.core.problem import Problem

#Algorithm
from pymoo.algorithms.moo.nsga2 import NSGA2

#Mixed variable 
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer
from pymoo.core.mixed import MixedVariableMating, MixedVariableSampling, MixedVariableDuplicateElimination

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

#CONSTRAINTS
bcrit = 5
wp = 200
buf_min = bcrit+1
buf_max = round(wp/2)
fact_min = 0
fact_max = 5

class MixedVarsInit(ElementwiseProblem):
    
    def __init__(self, **kwargs):
        
        variables = dict()
        
        #Declare the variable bounding conditions
        variables[f"x01"] = Integer(bounds=(buf_min, buf_max))
        variables[f"x02"] = Real(bounds=(fact_min, fact_max))
        variables[f"x03"] = Real(bounds=(fact_min, fact_max))

        super().__init__(vars=variables, n_obj=4, **kwargs)
        
    
    def _evaluate(self, designs, out, *args, **kwargs):

        #Declare the variables using the bounds above
        designs = np.array([designs[f"x01"],designs[f"x02"],designs[f"x03"]])
        res = []
        res.append(obj_wrapper(designs, *params))
        out['F'] = np.array(res)

#GA Settings
p_size = 200
g_size = 100

problem = MixedVarsInit()

#Declare the algorithm

algorithm = NSGA2(pop_size=p_size,sampling=MixedVariableSampling(),mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                  eliminate_duplicates=MixedVariableDuplicateElimination(),)

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