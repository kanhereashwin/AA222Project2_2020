#
# File: project2.py
#

## top-level submission file

'''
Note: Do not import any other modules here.
        To import from another file xyz.py here, type
        import project2_py.xyz
        However, do not import any modules except numpy in those files.
        It's ok to import modules only in files that are
        not imported here (e.g. for your plotting code).
'''
import numpy as np


def optimize(f, g, c, x0, n, count, prob):
    """
    Args:
        f (function): Function to be optimized
        g (function): Gradient function for `f`
        c (function): Function evaluating constraints
        x0 (np.array): Initial position to start from
        n (int): Number of evaluations allowed. Remember `f` and `c` cost 1 and `g` costs 2
        count (function): takes no arguments are reutrns current count
        prob (str): Name of the problem. So you can use a different strategy 
                 for each problem. `prob` can be `simple1`,`simple2`,`simple3`,
                 `secret1` or `secret2`
    Returns:
        x_best (np.array): best selection of variables found
    """
    _, x_best = optimize_history(f, g, c, x0, n, count, prob, debug=False)

    return x_best


class NormalizedGradientDescent:
    def __init__(self, optim_params):
        self.alpha = optim_params['alpha']
        self.f_eval = False
        self.g_eval = True

    def step(self, x, f_val = None, g_val = None):
        return x - self.alpha*g_val/np.linalg.norm(g_val)

class GradientDescent:
    def __init__(self, optim_params):
        self.alpha = optim_params['alpha']
        self.f_eval = False
        self.g_eval = True

    def step(self, x, f_val = None, g_val = None):
        return x - self.alpha*g_val

# class NoisyGradientDescent:
#     def __init__(self, optim_params):
#         self.alpha = optim_params['alpha']
#         self.f_eval = False
#         self.g_eval = True
#         self.random_step = optim_params['random_step']

#     def step(self, x, f_val = None, g_val = None):
#         rand_step = self.random_step*np.random.uniform(0., 1.)
#         return x - self.alpha*g_val + rand_step

class QuadraticPenalty:
    def __init__(self, penalty_param, optim_method, optim_params, problem, n, count_func):
        self.method = optim_method(optim_params)
        self.rho = penalty_param['rho']
        self.rho_step = 1
        self.f_eval = self.method.f_eval
        self.g_eval = self.method.g_eval
        self.c_eval = bool(True + self.f_eval) #CHECK
        self.grad_tol = penalty_param['grad_tol']
        self.x_tol = penalty_param['x_tol']
        self.step_size = penalty_param['step_size']
        self.xdim, self.cdim = find_dim(problem)
        self.n = n
        self.count_func = count_func

    def penalty_grad(self, c_func, x, c_val):
        # Assuming evaluating constraint returns numpy array
        # c_grad_est = est_gradient(c_func, x, c_val + 0.1, self.xdim, self.cdim) # CHECK
        c_grad_est = est_gradient(c_func, x, c_val, self.xdim, self.cdim) # CHECK
        # print('The point is ', x)
        # print('The constraint values at the point are ', c_val)
        # print('The estimated gradient is ')
        # print(c_grad_est)
        # input()
        # print('Multiplied by the violated constraints, the gradient is ')
        # print(np.dot(c_grad_est, c_val*np.asarray(c_val>0, dtype=int)))
        penalty_grad = (2*self.rho) *np.dot(c_grad_est, c_val*np.asarray(c_val>0, dtype=int))
        # print('The penalty gradient is ', penalty_grad)
        return penalty_grad

    def penalty_eval(self, c_val):
        # test = (1*self.rho)*np.sum(np.max(c_val, np.zeros_like(c_val))**2)
        # print('The value of the penalty is ', test)
        return (1*self.rho)*np.sum(np.max(c_val, np.zeros_like(c_val))**2)

    def step(self, x, f_val=None, g_val=None, c_val=None, c_func=None):
        if self.f_eval:
            aug_f = f_val + self.penalty_eval(c_val)
        else:
            aug_f = None
        if self.g_eval:
            pen_grad = self.penalty_grad(c_func, x, c_val)
            pen_grad_norm = np.linalg.norm(pen_grad)
            aug_g = g_val + pen_grad
        else:
            pen_grad_norm = 100000 # Arbitrarily large number
            aug_g = None
        # if np.mod(self.rho_step, 10) == 0:
        #     self.rho = 1.1*self.rho
        new_x = self.method.step(x, f_val=aug_f, g_val = aug_g)
        next_x = new_x
        delta_x = np.linalg.norm(new_x - x)
        self.rho_step += 1
        # print(c_val)
        # print(np.sum(np.asarray(c_val > 0, dtype = int)) !=0)
        # print(pen_grad_norm)
        # print(x)
        # input()
        if delta_x < self.x_tol or pen_grad_norm < self.grad_tol:
            if np.sum(np.asarray(c_val > 0, dtype = int)) !=0:
                next_x = findfeasible(x, self.step_size, c_func, self.count_func, self.n, self.xdim)       
        return next_x


class InteriorPoint:
    def __init__(self, penalty_param, optim_method, optim_params, problem, n, count_func):
        self.method = optim_method(optim_params)
        self.rho = penalty_param['rho']
        self.f_eval = self.method.f_eval
        self.g_eval = self.method.g_eval
        self.c_eval = bool(True + self.f_eval) #CHECK
        self.step_size = penalty_param['step_size']
        self.xdim, self.cdim = find_dim(problem)
        self.n = n
        self.problem = problem
        self.count_func = count_func
        self.feasible = False

    def penalty_grad(self, c_func, x, c_val):
        # Assuming evaluating constraint returns numpy array
        # c_grad_est = est_gradient(c_func, x, c_val + 0.1, self.xdim, self.cdim) # CHECK
        c_grad_est = est_gradient(c_func, x, c_val, self.xdim, self.cdim) # CHECK
        # print('The point is ', x)
        # print('The constraint values at the point are ', c_val)
        # print('The estimated gradient is ')
        # print(c_grad_est)
        # input()
        # print('Multiplied by the violated constraints, the gradient is ')
        # print(np.dot(c_grad_est, c_val*np.asarray(c_val>0, dtype=int)))
        penalty_grad = (2*self.rho) *np.dot(c_grad_est, (1/c_val)**2)
        # print('The penalty gradient is ', penalty_grad)
        return penalty_grad

    def penalty_eval(self, c_val):
        # test = (1*self.rho)*np.sum(np.max(c_val, np.zeros_like(c_val))**2)
        # print('The value of the penalty is ', test)
        return (self.rho)*np.sum(1/c_val)

    def step(self, x, f_val=None, g_val=None, c_val=None, c_func=None):
        if np.sum(np.asarray(c_val >0, dtype=int)) ==0:
            self.feasible = True
        else:
            self.feasible = False
        if self.feasible:
            # Execute optimization step
            if self.f_eval:
                aug_f = f_val + self.penalty_eval(c_val)
            else:
                aug_f = None
            if self.g_eval:
                pen_grad = self.penalty_grad(c_func, x, c_val)
                pen_grad_norm = np.linalg.norm(pen_grad)
                aug_g = g_val + pen_grad
            else:
                pen_grad_norm = 100000 # Arbitrarily large number
                aug_g = None
            next_x = self.method.step(x, f_val=aug_f, g_val = aug_g)
        else:
            # Execute random step
            if self.problem == 'simple2':
                next_x = self.step_size*np.random.randn(self.xdim) 
            else:
                next_x = self.step_size*np.random.randn(self.xdim)
        # if np.mod(self.rho_step, 10) == 0:
        #     self.rho = 1.1*self.rho
        # print(c_val)
        # print(np.sum(np.asarray(c_val > 0, dtype = int)) !=0)
        # print(pen_grad_norm)
        # print(x)
        # input()       
        return next_x
        

def est_gradient(func, x, c_val, x_dim, c_dim, h=1e-7):
    # CHECK: Might need complex step method for secret1
    grad_est = np.zeros([x_dim, c_dim])
    if np.sum(c_val > 0):
        for x_idx in range(x_dim):
            x_new = np.copy(x)
            x_new[x_idx] += h
            grad_est[x_idx, :] = (func(x_new) - c_val)/h
        # for idx in range(c_dim):
        #     print('For index ', idx, 'the gradient vector is ', grad_est[:, idx].T)
    return grad_est


def find_dim(problem):
    if problem == 'simple1':
        xdim = 2
        cdim = 2
    elif problem == 'simple2':
        xdim = 2        
        cdim = 2
    elif problem == 'simple3':
        xdim = 3
        cdim = 1
    elif problem == 'secret1':
        xdim = 60
        cdim = 105
    elif problem == 'secret2':
        xdim = 10
        cdim = 13
    elif problem == 'test':
            xdim = 2
    else:
        ValueError('Incorrect problem specified')
    return xdim, cdim

def complex_step_grad(f_func, x, xdim):
    grad_f = np.zeros(xdim)
    for x_idx in range(xdim):
        x_new = np.copy(x) + 0j
        x_new[x_idx] = 0 + 1e-7j
        grad_f[x_idx] = np.imag(f_func(x_new))/1e-7
    return grad_f

def findfeasible(x0, step_size, func_c, count_func, n, xdim):
    # CHECK
    num_pts_checked = 0
    # print('Finding feasible point')
    while True:
        # x = x0 + np.random.uniform(-step_size, step_size, xdim)
        # x = x0 + step_size*np.sign(np.random.uniform(-1, 1, xdim))
        x = x0 + step_size*np.random.randn(xdim)
        x = x0 + np.random.randn(xdim)
        num_pts_checked += 1
        # print(x)
        # input()
        c_val = func_c(x)
        if np.sum(np.asarray(c_val > 0, dtype = int)) ==0:
            # print('New point found')
            break
        if count_func() > 9*n/10 or count_func() > 1800:
            break
    #print(num_pts_checked)
    return x


def optimize_history(f, g, c, old_x, n, count, prob, debug=True):
    x_hist = []
    x_dim, _ = find_dim(prob)
    #old_x = findfeasible(c, count, n, prob)
    last_feasible_x = np.zeros(x_dim)
    # Define the parameters for the solver and the penalty
    penalty_params = {'rho' : 50, 'grad_tol': 1e-3, 'x_tol': 0.1, 'step_size' : 2} # Quadratic penalty
    penalty_params = {'rho' : 50, 'step_size' : 1} # Interior Point
    if prob == 'simple2':
        penalty_params['step_size'] = 2
        penalty_params['rho'] = 100
    optim_params = {'alpha' : 0.001, 'random_step' : 0} # Gradient Descent
    #optim = QuadraticPenalty(penalty_params, GradientDescent, optim_params, prob, n, count)
    optim = InteriorPoint(penalty_params, GradientDescent, optim_params, prob, n, count)
    if debug == True:
        print('Solving problem ', prob)
        print('Initial condition is', old_x)
        print('Initial function evaluation', f(old_x))
        print('Initial count ', count())
        print('Number of counts allowed', n)
    x_hist.append(old_x)
    while count() < n-100:
        if optim.f_eval:
            f_val = f(old_x)
        else:
            f_val = None
        if optim.c_eval:
            c_val = c(old_x)
        else:
            c_val = None
        if optim.g_eval:
            if prob == 'secret1':
                g_val = complex_step_grad(f, old_x, x_dim)
            else:
                g_val = g(old_x)
            #c_grad_val = est_gradient(c, old_x) CHECK Should be deprecated
        else:
            g_val = None
            #c_grad_val = None
        new_x = optim.step(old_x, f_val=f_val, g_val=g_val, c_val=c_val, c_func=c)
        x_hist.append(new_x)
        if np.sum(np.asarray(c_val >0, dtype = int)) == 0:
            last_feasible_x = np.copy(old_x)
        old_x = np.copy(new_x)
        #print('Point is ', old_x)
        #input()
        if debug:
            if count() > 2000:
                break
    if debug:
        print('Final count number ', count())
        print('Final function evaluation ', f(x_hist[-1]))
        print('Final x is ', x_hist[-1])
        print(optim.feasible)
        if prob == 'simple1':
            x_star = np.array([2/3, 1/np.sqrt(3)])
            print('Optimal x is ', x_star)
            print('Distance from optimal point is ', np.linalg.norm(x_star - new_x))
            print('Constraint values for feasability ', c(new_x))
        elif prob == 'simple2':
            x_star = np.array([1, 1])
            print('Optimal x is ', x_star)
            print('Distance from optimal point is ', np.linalg.norm(x_star - new_x))
            print('Constraint values for feasability ', c(new_x))
        elif prob == 'simple3':
            x_star = np.array([-1/np.sqrt(6), np.sqrt(2/3), -1/np.sqrt(6)])
            print('Optimal x is ', x_star)
            print('Distance from optimal point is ', np.linalg.norm(x_star - new_x))
            print('Constraint values for feasability ', c(new_x))
        elif prob == 'test':
            x_star = np.array([0, 0])
            print('Optimal x is ', x_star)
            print('Distance from optimal point is ', np.linalg.norm(x_star - new_x))
        else:
            print('Optimal point not entered yet')
    # print('Last feasible point ', last_feasible_x)
    if np.sum(np.asarray(c_val > 0, dtype = int)) !=0 and np.sum(last_feasible_x) != 0 and np.sum(last_feasible_x) != np.nan:
        x_hist.append(last_feasible_x)
    return x_hist, x_hist[-1]