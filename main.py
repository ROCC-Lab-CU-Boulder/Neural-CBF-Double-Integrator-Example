import numpy as np
import dataviz
import torch
from sys import argv

from system import System
from net import Model, ModelTrainer, prompt_save_model

weights_path = argv[2] if len(argv) > 1 and argv[1] == "--weights" else None
simulate_trajectories = False if weights_path is not None and "--sim" not in argv else True # only simulate if weights path is empty or if user requests to

# let's use double integrator
A = np.array([
    [0, 1],
    [0, 0],
])
B = np.array([
    0,
    1,
])

zeta = 0.7
omega = 1

K = np.array([omega**2, 2 * zeta * omega])

u_lb, u_ub = (-100, 100)
def u_π(x, v=None):
    # if reference is empty, treat as input without reference
    if v is None:
        res = np.dot(-K, x)
    # if reference is not empty, use it to shift the feedback
    else:
        res = np.dot(-K, x - np.array([1, 0]) * v)
    res = np.clip(res, u_lb, u_ub)
    return res

def f(x):
    x = np.asarray(x).flatten()
    return A @ x

def g(x): return B

tspan = [0,7]
bounds = [
    [-2, 2], # x1
    [-5, 5], # x2
]
n = len(bounds[0])

activations = ["gelu", "tanh", "elu"]
losses = ["mae", "huber", "logcosh"]

gather_data_on_hyperparams = False

######### FIXME: overhead crane dynamics ##############
def crane_stuff():
    def M(q):
        x, theta = q
        return np.array([
        [m_c + m_p, -m_p * L * np.cos(theta)],
        [-m_p * L * np.cos(theta), m_p * L**2],
    ])
    def V_m(q, qdot):
        x, theta = q
        xdot, thetadot = qdot
        return np.array([
            [0, m_p * L * thetadot * np.sin(theta)],
            [0, 0]
        ])
    def G(q):
        x, theta = q
        return np.array([
            [0],
            [m * g * L * np.sin(theta)],
        ])
    def F_crane(t, z):
        q = z[0:2]
        qdot = z[2:4]
        x, theta = q
        xdot, thetadot = qdot
        return np.array([
            x,
            theta,
            np.linalg.inv(M(q)) * (-V_m(q, qdot) * qdot - G(q) + B * u)
        ])
#######################################################

def F(t, x, v): return f(x) + g(x) * u_π(x, v)
if gather_data_on_hyperparams:
    for this_activation in activations:
        for this_loss in losses:
            system = System(
                F,
                bounds, tspan
            )
            model = Model(
                input_dim=n, # input to model is x (n)
                output_dim=2 * n,
                hidden_dim=128,
                activation=this_activation,
                dropout_rate=0.2
            )
            trainer = ModelTrainer(
                model,
                loss_type=this_loss,
                lr=1e-4,
                batch_size=128,
                epochs=500,
                plots=True
            )
            x_train, y_train, x_test, y_test = system.make_train_test_data()
            trainer.train(x_train, y_train)
            trainer.test(x_test, y_test)
    dataviz.compare_errors()
    dataviz.compare_losses()
    print("finished gathering hyperparameter perf data.")
    exit(0)
else:
    def F(t, x, v): return f(x) + g(x) * u_π(x)
    system = System(
        F,
        bounds,
        tspan,
        simulate_trajectories=simulate_trajectories,
    )
    model = Model(
        input_dim=n, # input to model is x (n)
        output_dim=2 * n,
        hidden_dim=128,
        activation=activations[0],
        dropout_rate=0.2
    )
    trainer = ModelTrainer(
        model,
        loss_type=losses[1],
        lr=1e-4,
        batch_size=128,
        epochs=500,
        plots=False,
    )
    if weights_path is not None:
        trainer.model.load_state_dict(torch.load(weights_path, weights_only=True))
        print(f"Loaded weights from {weights_path}. Skipping training...")
    else:
        x_train, y_train, x_test, y_test = system.make_train_test_data()
        trainer.train(x_train, y_train)
        trainer.test(x_test, y_test)

# take a controller that has a higher ω and find an initial
# condition such that that nominal controller will violate my constraints
# start the condition inside the level set of h(0)
omega = omega * 20
K = np.array([omega**2, 2 * zeta * omega])

# FIXME: choosing a starting reference and ending reference only works when
# dist(r,v) less than around 0.5

r = -1.0
def u_n(x):
    x = np.squeeze(x)
    return np.dot(-K, x - np.array([1, 0]) * r)

x0s = [
    np.array(x0)
    for x0 in
    [
        [-1.8, 0.0],
        [-1.6, 0.0],
        [-1.0, 0.0],
        [-0.5, 0.0],
        [1.5, 0.0],
    ]
]

if simulate_trajectories:
    system.plot_hs(trainer)
    system.plot_quivers_and_contours(trainer)

system.plot_cbf_trajectory(trainer, f, g, u_n, x0s, (u_lb, u_ub))

# if the user did not provide weights, then ask if they want to save the
# current ones
if weights_path is None:
    prompt_save_model(trainer)
