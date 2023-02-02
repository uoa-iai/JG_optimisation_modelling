#This script uses pymoo and NSGA2 to optimise our favourite system

from pymoo.optimize import minimize
from SA import *
import numpy as np
import pickle
from pymoo.core.callback import Callback

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
from pymoo.visualization.scatter import Scatter
from pymoo.util.display.column import Column
from pymoo.util.display.output import Output

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

#MODE SELECTION - omx OR tb3
mode = 'omx'

params = (lat_kde,lat_lan,p_bad,mode)

#CONSTRAINTS
bcrit = 5
wp = 1000
buf_min = bcrit+1
buf_max = round(wp/10)
fact_min = 0
fact_max = 5

class MyCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.data["best"] = []

    def notify(self, algorithm):
        self.data["best"].append(algorithm.pop.get("F").min())


class MixedVarsInit(ElementwiseProblem):
    
    def __init__(self, **kwargs):
        
        variables = dict()
        
        #Declare the variable bounding conditions
        variables[f"x01"] = Integer(bounds=(buf_min, buf_max))
        variables[f"x02"] = Real(bounds=(fact_min, fact_max))
        variables[f"x03"] = Real(bounds=(fact_min, fact_max))
        variables[f"x04"] = Real(bounds=(0.0, 1.0))

        super().__init__(vars=variables, n_obj=4, **kwargs)
        
    
    def _evaluate(self, designs, out, *args, **kwargs):

        #Declare the variables using the bounds above
        designs = np.array([designs[f"x01"],designs[f"x02"],designs[f"x03"],designs[f"x04"]])
        res = []
        res.append(obj_wrapper(designs, *params))
        out['F'] = np.array(res)
        
        
        

#GA Settings
p_size = 200
g_size = 50

problem = MixedVarsInit()

#Declare the algorithm

algorithm = NSGA2(pop_size=p_size,sampling=MixedVariableSampling(),mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                  eliminate_duplicates=MixedVariableDuplicateElimination(),)

stop_criteria = ('n_gen',g_size)

results = minimize(
    problem=problem,
    algorithm=algorithm,
    termination=stop_criteria,
    callback=MyCallback(),
    verbose=True
)
print(" POP: "+str(p_size)+"   "+str(g_size)+" Generations")
print("         FINISHED           ")
print(results.F)
print("             INPUTS        ")
print(results.X)

costFile = open('nsga_cost','wb')
pickle.dump(results.F,costFile)
costFile.close()

inputFile = open('nsga_input','wb')
pickle.dump(results.X,inputFile)
inputFile.close()

plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(results.F, facecolor="none", edgecolor="red")
plot.show()