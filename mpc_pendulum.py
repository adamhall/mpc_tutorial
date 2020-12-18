from casadi import *
import numpy as np
import matplotlib.pyplot as plt

# Solve inverted pendulum with Casadi nmpc



def setup_iter(f, N, Q, R):
    opti = casadi.Opti()
    X = opti.variable(2, N+1)
    U = opti.variable(1, N)
    P = opti.parameter(2, 2) # initial condition and goal location
    #opti.set_value(P[:,0], x0)
    #opti.set_value(P[:,1], xg)
    J = 0
    for i in range(N):
        J += mtimes((X[:, i]-P[:,1]).T, mtimes(Q, (X[:, i]-P[:,1]))) + mtimes(U[:, i].T, mtimes(R, U[:, i]))

    opti.minimize(J)
    for i in range(N):
        #opti.subject_to(X[:, i + 1] == f(X[0, i], X[1, i], U[i]))
        opti.subject_to(X[:, i + 1] == f(x0=X[:, i], p=U[i])['xf'])

    opti.subject_to(X[:, 0] == P[:, 0])

    opti.solver('ipopt')

    opti_dict = {'opti' : opti,
                 'X' : X,
                 'U' : U,
                 'P' : P}
    return opti_dict

def solve_iter(opti_dict, x0, xg):

    opti = opti_dict['opti']
    X = opti_dict['X']
    U = opti_dict['U']
    P = opti_dict['P']

    opti.set_value(P[:,0], x0)
    opti.set_value(P[:,1], xg)

    sol = opti.solve()

    return sol.value(X), sol.value(U)

if __name__ == "__main__":
    np.random.seed(123412)

    N = 10
    T = 10

    x1 = MX.sym('x1')
    x2 = MX.sym('x2')
    u = MX.sym('u')

    m = 1.0
    g = 9.8
    l = 1.0
    I = m*l**2/3.0
    dt = 0.01

    x_dot = vertcat(x2, -3*g/(2*l)*sin(x1) + u/I)
    fc = Function('fc', [x1, x2, u], [x_dot]) # continuous dynamics

    x = vertcat(x1, x2)
    dae = {'x': x, 'p': u, 'ode': x_dot}
    opts = {'tf': dt}
    F = integrator('F', 'cvodes', dae, opts)

    x1_lim = pi
    x2_lim = 1.0
    x_lim = np.array([x1_lim, x2_lim])
    x0 = 2*x_lim*np.random.random(2) - x_lim
    xg = np.zeros(2)
    Q = np.eye(2)*10
    R = np.eye(1)
    tol = 1e-3
    MAX_STEPS = 1000
    x = x0.reshape((2,1))
    step = 0
    opti_dict = setup_iter(F, N, Q, R)
    all_x = [x]
    all_u = []
    while( np.linalg.norm(x-xg) > tol and step < MAX_STEPS):
        X, U = solve_iter(opti_dict, x, xg)
        u = U[0]
        print('')
        print("Step: %s" % step)
        print('State: %0.3f  %0.3f,  Input: %0.3f' % (x[0], x[1], u))
        step += 1
        x = F(x0=x, p=u)['xf'].toarray()
        all_x += [x]
        all_u += [u]

all_x = np.array(all_x)
all_u = np.array(all_u)
N_steps = all_x.shape[0]
t = np.arange(N_steps)*dt
plt.plot(t, all_x[:,0])
plt.plot(t, all_x[:,1])
plt.plot(t[0:-1], all_u)
plt.legend(['Theta','Theta_dot','input'])
plt.show()




#x = SX.sym('x', 2)
#z = SX.sym('z')
#u = SX.sym('u')
#f = vertcat( z*x[0] - x[1]+u,
#             x[0])
#g = x[1]**2 + z - 1
#h = x[0]**2 + x[1]**2 + u**2
#dae = dict( x=x, p=u, ode=f, z=z, alg=g, quad=h)
#
#T = 10
#N = 10
#op = dict(t0=0, tf=T/N)
#F = integrator('F', 'idas', dae, op)
#
##empty nlp
#w = []
#lbw = []
#ubw = []
#G = []
#J = []
#
#Xk = MX.sym('X0', 2)
#w += [Xk]
#lbw += [0, 1]
#ubw += [0, 1]
#
#for k in range(1, N+1):
#    Uname = 'U' + str(k-1)
#    Uk = MX.sym(Uname)
#    w += [Uk]
#    lbw += [-1]
#    ubw += [1]
#
#    Fk = F(x0=Xk, p=Uk)
#    J += Fk['qf']
#
#    Xname = 'X' + str(k)
#    Xk = MX.sym(Xname, 2)
#    w += [Xk]
#    lbw += [-0.25, -inf]
#    ubw += [inf, inf]
#
#    G += [Fk['xf']-Xk]
#
#
#nlp = dict(f=J, g=vertcat(*G), x=vertcat(*w))
#S = nlpsol('S', 'ipopt', nlp)
#
#r = S(lbx=lbw, ubx=ubw, x0=0, lbg=0, ubg=0)
#print(r['x'])
#
#
#
