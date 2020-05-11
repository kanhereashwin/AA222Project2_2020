import numpy as np
import project2
import plotting
from helpers import Simple1, Simple2, Simple3

test_problem = Simple2()
debug_set = False
# if debug_set:
#     test_problem.nolimit()
# for i in range(1):
#     optim_history, best= project2.optimize_history(test_problem.f, test_problem.g, test_problem.c, test_problem.x0(), test_problem.n, test_problem.count, test_problem.prob, debug=debug_set)
#     test_problem._reset()
plotting.plot_history()
#plotting.plot_traject()