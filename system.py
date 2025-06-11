import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import itertools
import traceback
import time

from tqdm import tqdm
from datetime import datetime
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from net import ModelTrainer
from optim import solve_controller
from scipy.integrate import solve_ivp
from scipy.interpolate import griddata, interp1d
from typing import Callable, List, Tuple, Dict, Union
from multiprocessing import Pool
from functools import partial


def _compute_max_v_and_trajectories(x0, vs, tvec, f, constraints, n):
    """Compute max_v min_t x(t) given x0."""
    max_v = None
    xs = []
    for v in vs:
        sol = solve_ivp(lambda t, x: f(t, x, v), [tvec[0], tvec[-1]], x0, t_eval=tvec)
        xs.append(sol)
        min_ts = np.array([
            min_t
            for i in range(n)
            for min_t in [
                np.min(-constraints[i][0] + sol.y[i]),
                np.min(constraints[i][1] - sol.y[i]),
            ]
        ])
        if max_v is not None:
            max_v = np.maximum(max_v, min_ts)
        else:
            max_v = min_ts
    return max_v, xs


class System:
    # The number divisions we divide the grid in which our initial conditions exist.
    n_subdivisions = 20

    def __init__(
        self,
        f: Callable,
        constraints: List[Tuple[float, float]],
        tspan: List,
        simulate_trajectories: bool = True,
        dt: int = 0.01,
    ):
        """
        Initialize the system.

        Parameters:
        -----------
        f : Callable
        constraints : List[Tuple]
            A list of tuples [(lb, ub), (lb, ub), ...] where lb and ub
            represent lower and upper bounds.
        simulate_trajectories : bool
            Flag to specify whether or not trajectories and h values should be simulated.
        tspan : List[float, float]
            Time for which to simulate the system.
        """
        assert f.__name__ != "<lambda>", "`f` cannot be a lambda function. Use a statically defined function instead."
        n = len(constraints[0])
        self.n = n
        self.constraints = constraints
        self.dt = dt
        self.tvec = np.linspace(tspan[0], tspan[1], int((tspan[1] - tspan[0]) / self.dt))

        # add a little extra onto the simulated initial positions for better training
        x0s_grid = np.meshgrid(*[np.linspace(c[0] - abs(c[0]) * 0.5, c[1] + abs(c[1]) * 0.5, self.n_subdivisions) for c in self.constraints])
        self.x0s_x1_grid, self.x0s_x2_grid = x0s_grid
        # simulate the system from each initial condition sampled between each bound
        self.x0s = np.array(x0s_grid).T.reshape(-1, n)
        self.xs = []
        # select first dimension across all unique x0s
        self.vs = np.unique(self.x0s[:, 0])

        if simulate_trajectories:
            # gather hs
            # h_i(x) = max_{v} min_{t} c_i(x(t) | x0,v)
            max_vs = []
            partial_applied = partial(
                _compute_max_v_and_trajectories,
                vs=self.vs,
                tvec=self.tvec,
                f=f,
                constraints=constraints,
                n=self.n,
            )
            with Pool() as pool:
                # use pool.imap_unordered to promote faster time to finish and lazy dumping of progress
                results: List[Tuple] = tqdm(pool.imap_unordered(partial_applied, self.x0s), desc="Simulating Trajectories", total=len(self.x0s))
                max_vs, xs_per_v = zip(*results)
                # want shape to be (4, len(self.x0s))
                self.hs = np.array(max_vs).T
                # flatten the list of xs_per_v
                self.xs = [x for xs in xs_per_v for x in xs]


    def make_train_test_data(self) -> Tuple[torch.tensor, torch.tensor]:
        """Generate the training data."""
        # how much of the total data goes to training. The rest goes to testing.
        split_ratio = 0.80
        n_samples = len(self.x0s)
        indices = list(range(n_samples))
        random.shuffle(indices)
        split_idx = int(n_samples * split_ratio)
        train_indices = torch.tensor(indices[:split_idx])
        test_indices = torch.tensor(indices[split_idx:])

        inputs_tensor = torch.tensor(self.x0s, dtype=torch.float32)
        hs_tensor = torch.tensor(self.hs, dtype=torch.float32).T
        x_train = inputs_tensor[train_indices]
        y_train = hs_tensor[train_indices]
        x_test = inputs_tensor[test_indices]
        y_test = hs_tensor[test_indices]
        return x_train, y_train, x_test, y_test

    def plot_cbf_trajectory_setup(self, model_trainer: ModelTrainer):
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["mathtext.fontset"] = "dejavuserif"

        fig, ax = plt.subplots(figsize=(6, 6))

        #####################
        # Plot the safe set #
        #####################
        safe_rect = Rectangle((self.constraints[0][0], self.constraints[1][0]),
                            self.constraints[0][1] - self.constraints[0][0],
                            self.constraints[1][1] - self.constraints[1][0],
                            linewidth=2, edgecolor='green', facecolor='none')
        ax.add_patch(safe_rect)

        # Fill the outside regions with transparent red + hatching
        x1_min, x1_max = self.constraints[0]
        x2_min, x2_max = self.constraints[1]

        # Define the limits of the entire plot
        margin = 2
        xlim = [x1_min - margin, x1_max + margin]
        ylim = [x2_min - margin, x2_max + margin]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # Fill each of the 4 outside regions
        # Bottom
        ax.fill_betweenx([ylim[0], x2_min], xlim[0], xlim[1], color='red', alpha=0.2, hatch='//')
        # Top
        ax.fill_betweenx([x2_max, ylim[1]], xlim[0], xlim[1], color='red', alpha=0.2, hatch='//')
        # Left
        ax.fill_betweenx([x2_min, x2_max], xlim[0], x1_min, color='red', alpha=0.2, hatch='//')
        # Right
        ax.fill_betweenx([x2_min, x2_max], x1_max, xlim[1], color='red', alpha=0.2, hatch='//')

        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title(f"Nominal trajectory versus CBF-controlled trajectory \n ({model_trainer.loss_type}+{model_trainer.model.activation_type})")
        ax.grid(True)

        return (fig, ax)

    def plot_cbf_trajectory(self, model_trainer: ModelTrainer, f, g, u_nom, x0s: np.ndarray, u_bounds: Tuple, show_optim_results=True):
        """Plot trajectories of safe trajectories using the CBF versus nominal trajectories."""
        fig, ax = self.plot_cbf_trajectory_setup(model_trainer=model_trainer)

        grad_h = model_trainer.compute_input_gradients
        h = model_trainer.predict
        # create list of control inputs derived from the CBF
        u_noms = []
        u_cbfs = []
        trajectories_nom = []
        trajectories_cbf = []

        def u_cbf(x):
            return solve_controller(grad_h, h, f, g, u_nom, x, u_bounds)

        i = 0
        for x0 in x0s:
            try:
                print(f"{i=}")
                i += 1

                #####################
                # Plot trajectories #
                #####################
                # Nominal trajectory
                def solve_nom_traj():
                    def fun_nom(t, x):
                        u = u_nom(x)
                        return f(x) + g(x) * u
                    return solve_ivp(fun_nom, [self.tvec[0], self.tvec[-1]], x0, t_eval=self.tvec)

                # trajectory derived from CBF
                def solve_cbf_traj():
                    def fun_cbf(t, x):
                        u = u_cbf(x)
                        return f(x) + g(x) * u
                    return solve_ivp(fun_cbf, [self.tvec[0], self.tvec[-1]], x0, t_eval=self.tvec)

                nom_traj = solve_nom_traj()
                cbf_traj = solve_cbf_traj()

                u_noms.append(np.array([u_nom(x) for x in nom_traj.y.T]))
                u_cbfs.append(u_cbf(cbf_traj.y))

                ax.plot(nom_traj.y[0], nom_traj.y[1], label="$x(t)$ from $u_n(t)$", color='blue', linewidth=2)
                ax.plot(cbf_traj.y[0], cbf_traj.y[1], label="$x(t)$ from CBF", color='orange', linewidth=2)

                trajectories_nom.append(nom_traj.y)
                trajectories_cbf.append(cbf_traj.y)

                # plot contour plot where CBF is valid
                levels = [0,]
                min_hhats = np.min(model_trainer.predict(self.x0s), axis=1)
                min_hhats_grid = griddata((self.x0s[:, 0], self.x0s[:, 1]), min_hhats[:], (self.x0s_x1_grid, self.x0s_x2_grid), method="cubic")
                contourf = ax.contour(self.x0s_x1_grid, self.x0s_x2_grid, min_hhats_grid, levels=levels)
            except Exception as e:
                print(f"Exception occured while solving for {x0=}: {e}\n{traceback.format_exc()}")

        handles, labels = ax.get_legend_handles_labels()
        # Use a dictionary to remove duplicates
        by_label = dict(zip(labels, handles))
        # Create legend
        ax.legend(by_label.values(), by_label.keys())
        timestamp = datetime.now().strftime("%Y-%m-%dT%H.%M")
        filename = f"cbf_trajectory_{timestamp}_{model_trainer.loss_type}+{model_trainer.model.activation_type}.png"
        fig.savefig(filename)
        plt.show()

        for traj_nom in trajectories_nom:
            plt.plot(self.tvec, traj_nom[0], label="nominal") # plot x1
        for traj_cbf in trajectories_cbf:
            plt.plot(self.tvec, traj_cbf[0], label="cbf") # plot x1
        plt.grid(True)
        plt.legend()
        plt.show()

        if show_optim_results:
            for u_noms_i in u_noms:
                plt.plot(self.tvec, u_noms_i, label="$u_n(x)$ (nominal trajectory)")
            for u_cbfs_i in u_cbfs:
                plt.plot(self.tvec, u_cbfs_i, label="$u(x)$ (from CBF)")
            plt.grid(True)
            plt.legend()
            plt.show()

    def plot_cbf_trajectories_gif(self, model_trainer: ModelTrainer, f, g, u_nom, w_nom, x0s_and_vs: List[Tuple[np.ndarray, np.ndarray]], u_bounds: Tuple, show_optim_results=True):
        """Plot trajectories of safe trajectories using the CBF versus nominal trajectories as a GIF."""
        fig, ax = self.plot_cbf_trajectory_setup()

        # Initialize lines
        (nom_line,) = ax.plot([], [], label="$x(t)$ from $u_n(t)$", color='blue', linewidth=3)
        (cbf_line,) = ax.plot([], [], label="$x(t)$ from CBF", color='orange', linewidth=3)
        ax.legend()

        def init():
            nom_line.set_data([], [])
            cbf_line.set_data([], [])
            return nom_line, cbf_line

        grad_h = model_trainer.compute_input_gradients
        h = model_trainer.predict

        def u_cbf(z):
            x = z[0:self.n]
            v = z[self.n:]
            return solve_controller(grad_h, h, f, g, u_nom, x, w_nom, v, u_bounds)

        def update(frame):
            #####################
            # Plot trajectories #
            #####################
            x0, v = x0s_and_vs[frame]

            # Nominal trajectory
            def solve_nom_traj():
                def fun_nom(t, x): return np.atleast_2d(f(x)) + g(x) @ u_nom(x)
                return solve_ivp(fun_nom, [self.tvec[0], self.tvec[-1]], x0, t_eval=self.tvec)

            def solve_cbf_traj():
                def fun_cbf(t, x): return np.atleast_2d(f(x)) + g(x) @ u_cbf(x)
                return solve_ivp(fun_cbf, [self.tvec[0], self.tvec[-1]], x0, t_eval=self.tvec)

            sol_nom = solve_nom_traj()
            sol_cbf = solve_cbf_traj()

            # Update plots
            nom_line.set_data(sol_nom.y[0], sol_nom.y[1])
            cbf_line.set_data(sol_cbf.y[0], sol_cbf.y[1])
            print(f"Frame {frame + 1}/{len(x0s)} completed")
            return nom_line, cbf_line

        interval = 500 # ms
        ani = FuncAnimation(
            fig, update, init_func=init,
            frames=len(x0s), interval=interval,
            blit=True, repeat=False
        )
        # save the trajectories as a gif
        timestamp = datetime.now().strftime("%Y-%m-%dT%H.%M")
        filename = f"cbf_trajectories_{timestamp}.gif"
        ani.save(filename, writer="pillow", fps=1_000 // interval)

    def plot_hs(self, model_trainer=None):
        """Plot each h. This only works if n == 2."""
        if self.n != 2:
            raise ValueError(f"Dimensions of system are not suitable. {n=} when n=2 is needed.")

        plt.rcParams["font.family"] = "serif"
        plt.rcParams["mathtext.fontset"] = "dejavuserif"

        fig = plt.figure(figsize=(12, 10))
        nrows = self.n
        ncols = 2

        if model_trainer is None:
            nrows = self.n
            ncols = 2

            for i in range(len(self.hs)):
                ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d', title="$h_{" + str(i + 1) + "}(x)$")
                h_grid = griddata((self.x0s[:, 0], self.x0s[:, 1]), self.hs[i], (self.x0s_x1_grid, self.x0s_x2_grid), method="cubic")
                ax.plot_surface(self.x0s_x1_grid, self.x0s_x2_grid, h_grid, cmap="viridis")

            plt.tight_layout()
            plt.show()
        else:
            for i in range(len(self.hs)):
                ax = fig.add_subplot(nrows, ncols, i+1, projection='3d', title="$h_{" + str(i + 1) + "}(x)$")
                h_grid = griddata((self.x0s[:, 0], self.x0s[:, 1]), model_trainer.predict(self.x0s)[:, i], (self.x0s_x1_grid, self.x0s_x2_grid), method="cubic")
                ax.plot_surface(self.x0s_x1_grid, self.x0s_x2_grid, h_grid, cmap="viridis")
            # FIXME: how do I plot the unified plot of h?

            plt.show()


    def plot_xs(self):
        """Plot each trajectory of the system starting from the initial conditions generated."""

        plt.rcParams["font.family"] = "serif"
        plt.rcParams["mathtext.fontset"] = "dejavuserif"

        for traj in self.xs:
            plt.plot(traj.y[0], traj.y[1])
            plt.show()

    def plot_quivers_and_contours(self, model_trainer: ModelTrainer):
        """Plot quiver plots and contours for each trained h."""
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["mathtext.fontset"] = "dejavuserif"

        nrows = 2
        ncols = 2

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        hhats = [model_trainer.predict(self.x0s),]

        grad_hhats = [model_trainer.compute_input_gradients(self.x0s),]

        # levels for contour plot
        levels = np.arange(-5, 3)

        def init():
            for i, ax in enumerate(axes):
                ax.set_title("$\\hat{h}_{" + str(i + 1) + "}(x)$ and $\\nabla\\hat{h}_{" + str(i + 1) + "}(x)$")
                ax.grid(True)
            return axes

        cbars = []
        def update(frame):
            for cbar in cbars:
                cbar.remove()
            cbars.clear()
            for i, ax in enumerate(axes):
                ax.clear()
                ax.set_title("$\\hat{h}_{" + str(i + 1) + "}(x)$ and $\\nabla\\hat{h}_{" + str(i + 1) + "}(x)$")
                # plot the reference
                hhat_grid = griddata((self.x0s[:, 0], self.x0s[:, 1]), hhats[frame][:, i], (self.x0s_x1_grid, self.x0s_x2_grid), method="cubic")
                contourf = ax.contourf(self.x0s_x1_grid, self.x0s_x2_grid, hhat_grid, levels=levels, extend="max")
                cbar = fig.colorbar(contourf, ax=ax)
                cbar.set_ticks(levels)
                cbars.append(cbar)
                grad_x1_hhat_grid = griddata((self.x0s[:, 0], self.x0s[:, 1]), grad_hhats[frame][:, i, 0], (self.x0s_x1_grid, self.x0s_x2_grid), method="cubic")
                grad_x2_hhat_grid = griddata((self.x0s[:, 0], self.x0s[:, 1]), grad_hhats[frame][:, i, 1], (self.x0s_x1_grid, self.x0s_x2_grid), method="cubic")
                # coarsen the grid with a step value
                step = 3  # (higher -> coarser)
                ax.quiver(
                    self.x0s_x1_grid[::step, ::step], self.x0s_x2_grid[::step, ::step],
                    grad_x1_hhat_grid[::step, ::step], grad_x2_hhat_grid[::step, ::step],
                    scale=15,  # smaller -> bigger arrows
                    width=0.007 # thicker arrows
                )
            return axes

        ani = FuncAnimation(fig, update, frames=1, init_func=init, blit=False, repeat=True)
        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        def init():
            ax.set_title("$\\min_i\\  \\hat{h}_i(x|x_0)$")
            ax.set_xlim(*list(np.array(self.constraints[0]) * 1.2))
            ax.set_ylim(*list(np.array(self.constraints[1]) * 1.2))
            ax.grid(True)

        cbars = []
        def update(frame):
            for cbar in cbars:
                cbar.remove()
            cbars.clear()
            ax.clear()

            # plot the reference
            min_hhats = np.min(hhats[frame], axis=1)
            min_hhats_grid = griddata((self.x0s[:, 0], self.x0s[:, 1]), min_hhats[:], (self.x0s_x1_grid, self.x0s_x2_grid), method="cubic")
            contourf = ax.contourf(self.x0s_x1_grid, self.x0s_x2_grid, min_hhats_grid, levels=levels, extend="max")
            ax.contour(self.x0s_x1_grid, self.x0s_x2_grid, min_hhats_grid, levels=[0])
            cbar = fig.colorbar(contourf, ax=ax)
            cbar.set_ticks(levels)
            cbars.append(cbar)

        ani = FuncAnimation(fig, update, frames=1, init_func=init, blit=False, repeat=True)
        plt.show()
