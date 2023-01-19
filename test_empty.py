import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer, Choice, Binary

class MixedVarsZDT1(ElementwiseProblem):

    def __init__(self, **kwargs):

        variables = dict()

        variables[f"x01"] = Real(bounds=(0.0, 1.0))

        for k in range(2, 6):
            variables[f"x{k:02}"] = Real(bounds=(0.0, 10.0))

        for k in range(6, 11):
            variables[f"x{k:02}"] = Integer(bounds=(0, 10))

        super().__init__(vars=variables, n_obj=2, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.array([x[f"x{k:02}"] for k in range(1, 11)])

        f1 = x[0]
        g = 1 + 9.0 / (self.n_var - 1) * np.sum(x[1:])
        f2 = g * (1 - np.power((f1 / g), 0.5))
        out["F"] = [f1, f2]
        
problem = MixedVarsZDT1()


for name, var in problem.vars.items():
    print(name, var)