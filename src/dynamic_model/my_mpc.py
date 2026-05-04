import numpy as np
import sys
from casadi import *
import do_mpc

hidden_size=50
input_size=4
output_size=4

model_type = 'discrete' # either 'discrete' or 'continuous'
model = do_mpc.model.Model(model_type)

_x = model.set_variable(var_type='_x', var_name='x', shape=(hidden_size,1))
_u = model.set_variable(var_type='_u', var_name='u', shape=(input_size,1))

Wrec=np.abs(np.random.randn(hidden_size,hidden_size))
Wrec_bias=np.abs(np.random.randn(hidden_size,1))

B = np.abs(np.random.randn(hidden_size,input_size))
B_bias= np.abs(np.random.randn(hidden_size,1))

C=np.abs(np.random.randn(output_size,hidden_size))
C_bias=np.abs(np.random.randn(output_size,1))

h_next = (Wrec @ fmax(_x,0)) + Wrec_bias + B@_u + B_bias

model.set_rhs('x', h_next)

# Build the model
model.setup()

mpc = do_mpc.controller.MPC(model)


setup_mpc = {
    'n_robust': 0,
    'n_horizon': 4,
    't_step': 0.25,
    'state_discretization': 'discrete',
    'store_full_solution':True,
    # Use MA27 linear solver in ipopt for faster calculations:
    'nlpsol_opts': {'ipopt.linear_solver': 'mumps','ipopt.max_iter':100,'ipopt.print_level':0, 'ipopt.sb': 'yes', 'print_time':1} # 还没拿到MA27的license
}

mpc.set_param(**setup_mpc)

# objective function
# _x = model.x
design_ref=np.abs(np.random.randn(output_size,1))
mterm = norm_2(fmax(C@_x+C_bias,0)-design_ref) # terminal cost
lterm = norm_2(fmax(C@_x+C_bias,0)-design_ref) # terminal cost

# stage cost
mpc.set_objective(mterm=mterm, lterm=lterm)

mpc.set_rterm(u=1e-3) # input penalty


# bound of inputs
max_u = np.array([[20.0],[9.0],[20.0],[9.0]])
min_u = np.array([[0.0],[0.0],[0.0],[0.0]])
# lower bounds of the input
mpc.bounds['lower','_u','u'] = min_u
mpc.bounds['upper','_u','u'] = max_u

mpc.setup()


e = np.ones([model.n_x,1])
x0 = np.random.uniform(-3*e,3*e)
# for k in range(50):
import time

sum_time=0
for i in range(200):
    e = np.ones([model.n_x,1])
    x0 = np.random.uniform(-3*e,3*e)
    start = time.time()
    # given an initial state and output the optimal input
    u0 = mpc.make_step(x0)
    end = time.time()
    sum_time+=(end-start)*1000
    print('time_consumed:' , (end-start)*1000,'ms')
    print(u0)
print(sum_time/200)