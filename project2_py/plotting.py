import numpy as np
from matplotlib import pyplot as plt
from project2 import optimize_history
from helpers import Simple1, Simple2, Simple3

def plot_history():
    for p_type in [Simple2]:
        problem = p_type()
        init_x = problem.x0()
        x_hist_1, _ = optimize_history(problem.f, problem.g, problem.c, init_x, problem.n, problem.count, problem.prob, debug=False)
        problem = p_type()
        init_x = problem.x0()
        x_hist_2, _ = optimize_history(problem.f, problem.g, problem.c, init_x, problem.n, problem.count, problem.prob, debug=False)
        problem = p_type()
        init_x = problem.x0()
        x_hist_3, _ = optimize_history(problem.f, problem.g, problem.c, init_x, problem.n, problem.count, problem.prob, debug=False)
        problem.nolimit()
        obj_vals_1 = np.empty(len(x_hist_1))
        constraint_vals_1 = np.empty([problem._cdim, len(x_hist_1)])
        obj_vals_2 = np.empty(len(x_hist_2))
        constraint_vals_2 = np.empty([problem._cdim, len(x_hist_2)])
        obj_vals_3 = np.empty(len(x_hist_3))
        constraint_vals_3 = np.empty([problem._cdim, len(x_hist_3)])
        for pt_idx, pt in enumerate(x_hist_1):
            obj_vals_1[pt_idx] = problem.f(pt)
            constraint_vals_1[:, pt_idx] = problem.c(pt)
        for pt_idx, pt in enumerate(x_hist_2):
            obj_vals_2[pt_idx] = problem.f(pt)
            constraint_vals_2[:, pt_idx] = problem.c(pt)
        for pt_idx, pt in enumerate(x_hist_3):
            obj_vals_3[pt_idx] = problem.f(pt)
            constraint_vals_3[:, pt_idx] = problem.c(pt)
        plt.figure()
        plt.plot(obj_vals_1)
        plt.plot(obj_vals_2)
        plt.plot(obj_vals_3)
        plt.plot(np.zeros_like(obj_vals_1), label='Optimal objective value')
        plt.title('Convergence plot for ' + problem.prob + ' with initial point ' + str(init_x))
        plt.xlabel('Number of iterations')
        plt.ylabel('f(x)')
        plt.legend()
        plt.figure()
        for i in range(problem._cdim):
            plt.plot(constraint_vals_1[i, :], '-b')
            plt.plot(constraint_vals_2[i, :], '-r')
            plt.plot(constraint_vals_3[i, :], '-k')
        plt.plot(np.zeros_like(obj_vals_1), label='Boundary constraint value')
        plt.title('Constraint plot for ' + problem.prob + ' with initial point ' + str(init_x))
        plt.xlabel('Number of iterations')
        plt.ylabel('c(x)')
        plt.legend()
    plt.show()
    return None


def plot_traject():
    problem = Simple1()
    print('Simple1: 1st Trajectory')
    x_hist_1, _ = optimize_history(problem.f, problem.g, problem.c, problem.x0(), problem.n, problem.count, problem.prob, debug=False)
    print('Simple1: 2nd Trajectory')
    problem = Simple1()
    x_hist_2, _ = optimize_history(problem.f, problem.g, problem.c, problem.x0(), problem.n, problem.count, problem.prob, debug=False)
    print('Simple1: 3rd Trajectory')
    problem = Simple1()
    x_hist_3, _ = optimize_history(problem.f, problem.g, problem.c, problem.x0(), problem.n, problem.count, problem.prob, debug=False)
    problem.nolimit()
    x0_list = np.linspace(-3, 3, 101)
    x1_list = np.linspace(-3, 3, 101)
    X0, X1 = np.meshgrid(x0_list, x1_list)
    Z1 = simple1_eval(X0, X1)
    C11 = simple1_c1(X0, X1)
    C12 = simple1_c2(X0, X1)
    C1 = C11*C12
    x_hist_1 = np.array(x_hist_1)
    x_hist_2 = np.array(x_hist_2)
    x_hist_3 = np.array(x_hist_3)
    plt.figure()
    plt.contour(X0, X1, Z1, 10)
    plt.plot(x_hist_1[:, 0], x_hist_1[:, 1], '-b')
    plt.plot(x_hist_2[:, 0], x_hist_2[:, 1], '-r')
    plt.plot(x_hist_3[:, 0], x_hist_3[:, 1], '-k')
    plt.scatter(X0, X1, s=C1)
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])


    # Repeat above procedure for Simple2

    problem = Simple2()    
    x_hist_2_1, _ = optimize_history(problem.f, problem.g, problem.c, problem.x0(), problem.n, problem.count, problem.prob, debug=False)
    print('Simple2: 1st Trajectory')
    problem = Simple2()
    x_hist_2_2, _ = optimize_history(problem.f, problem.g, problem.c, problem.x0(), problem.n, problem.count, problem.prob, debug=False)
    print('Simple2: 2nd Trajectory')
    problem = Simple2()
    x_hist_2_3, _ = optimize_history(problem.f, problem.g, problem.c, problem.x0(), problem.n, problem.count, problem.prob, debug=False)
    print('Simple2: 3rd Trajectory')
    problem.nolimit()
    x0_list = np.linspace(-3, 3, 100)
    x1_list = np.linspace(-3, 3, 100)
    X0, X1 = np.meshgrid(x0_list, x1_list)
    Z2 = simple2_eval(X0, X1)
    C21 = simple2_c1(X0, X1)
    C22 = simple2_c2(X0, X1)
    C2 = C21*C22
    x_hist_2_1 = np.array(x_hist_2_1)
    x_hist_2_2 = np.array(x_hist_2_2)
    x_hist_2_3 = np.array(x_hist_2_3)


    plt.figure()
    plt.contour(X0, X1, Z2, 10)
    plt.plot(x_hist_2_1[:, 0], x_hist_2_1[:, 1], '-b')
    plt.plot(x_hist_2_2[:, 0], x_hist_2_2[:, 1], '-r')
    plt.plot(x_hist_2_3[:, 0], x_hist_2_3[:, 1], '-k')
    plt.scatter(X0, X1, s=C2)
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])

    plt.show()
    return None


def simple1_eval(X0, X1):
    return -X0 * X1  + 2.0 / (3.0 * np.sqrt(3.0))

def simple1_c1(X0, X1):
    C = X0 + X1**2 - 1
    C[C>0] = 0
    C[C<0] = 1
    return C

def simple1_c2(X0, X1):
    C = -X0 -X1
    C[C>0] = 0
    C[C<0] = 1
    return C


def simple2_eval(X0, X1):
    return 100 * (X1 - X0**2)**2 + (1-X0)**2

def simple2_c1(X0, X1):
    C = (X0-1)**3 - X1 + 1
    C[C>0] = 0
    C[C<0] = 1
    return C

def simple2_c2(X0, X1):
    C = X0 + X1 - 2 
    C[C>0.] = 0
    C[C<0] = 1
    return C

def test_c1(X0, X1):
    C = X0 + 0.001
    C[C>0] = 0
    C[C<0] = 1
    return C

def test_c2(X0, X1):
    C = X1 
    C[C>0] = 0
    C[C<0] = 1
    return C