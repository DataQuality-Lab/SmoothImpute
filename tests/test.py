import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from smoothimpute.imputers import Imputer
from smoothimpute.simulators import Simulator
from smoothimpute.evaluator import Evaluator
from smoothimpute.advisor import Advisor

# data = [[0, np.nan, 1],
#         [np.nan, 1, 0]]

data = np.random.rand(5, 3)
data = pd.DataFrame(data)

print(data)
simulator = Simulator(mcar_p=0.2, mar_p=0.1, mnar_p=0.1)
xmiss = simulator.simulate(data)
print(xmiss)

imputer = Imputer("mice")
data_filled = imputer.impute(xmiss)
print(data_filled)

evaluator = Evaluator()
result = evaluator.evaluate(xmiss, data_filled, data)
print(result)

evaluator = Evaluator("text")
result = evaluator.evaluate(xmiss, data_filled, data)
print(result)

advisor_c = Advisor()
result = advisor_c.advise("What is the main component in SmoothImpute")
print(result)
