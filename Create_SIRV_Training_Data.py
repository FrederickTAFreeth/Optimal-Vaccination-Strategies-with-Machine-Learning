# ---- Importing Libraries ----
import numpy as np
import tqdm
import matplotlib.pyplot as plt


# ---- Global Parameters ----
NUM_RUNS = 64 # 65536 # Number of epidemic trajectories
num_age_groups = 6 # Number of age groups
compartment_colours = ["green", "red", "black", "blue"]
opacities = np.linspace(start=0.1, stop=1.0, num=num_age_groups)
PROGRESS_BAR_FORMAT = "{desc}: {n_fmt}/{total_fmt} {unit} ({percentage:3.1f}%) | Elapsed: {elapsed} | Remaining: {remaining} | {rate_fmt}"

def SIRV(t_max, betas, alpha, gamma, delta, theta, rho, kappa, tau, plot_run=False):
    """
    Simulate a batch of stochastic age-structured SIRV epidemics.

    Array shapes:
        S, I, R, V: (NUM_RUNS, t_max, num_age_groups)
        betas: (num_age_groups, num_age_groups)
        Other model parameters: (num_age_groups,)

    Returns
        S, I, R, V: Stochastic compartment trajectories.
    """
    rng = np.random.default_rng(314159)
    betas = np.asarray(betas, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    delta = np.asarray(delta, dtype=float)
    theta = np.asarray(theta, dtype=float)
    rho = np.asarray(rho, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    tau = np.asarray(tau, dtype=float)

    # ---- Check Parameter Shapes ----
    if betas.shape != (num_age_groups, num_age_groups):
        raise ValueError(f"betas must have shape ({num_age_groups}, {num_age_groups})")

    for parameter, parameter_name in [
        (alpha, "alpha"),
        (gamma, "gamma"),
        (delta, "delta"),
        (theta, "theta"),
        (rho, "rho"),
        (kappa, "kappa"),
        (tau, "tau")
    ]:
        if parameter.shape != (num_age_groups,):
            raise ValueError(f"{parameter_name} must have shape ({num_age_groups},)")

        if np.any(parameter < 0):
            raise ValueError(f"{parameter_name} cannot contain negative rates")

    if theta[-1] != 0:
        raise ValueError("The final age group must have an aging rate of zero")

    def sample_routes(n, rates):
        total_rate = rates.sum(axis=-1, keepdims=True)
        p = np.divide(
            rates, total_rate,
            out=np.zeros_like(rates),
            where=total_rate > 0
        ) * (-np.expm1(-total_rate))
        p = np.concatenate([p, 1.0 - p.sum(axis=-1, keepdims=True)], axis=-1)
        return rng.multinomial(n=n, pvals=p)

    # ---- Initialise State Arrays ----
    S = np.zeros((NUM_RUNS, t_max, num_age_groups), dtype=int)
    I = np.zeros((NUM_RUNS, t_max, num_age_groups), dtype=int)
    R = np.zeros((NUM_RUNS, t_max, num_age_groups), dtype=int)
    V = np.zeros((NUM_RUNS, t_max, num_age_groups), dtype=int)
    N_t0 = np.array([1060, 1265, 1642, 4857, 3312, 2477], dtype=int) * 10**3
    S[:, 0, :] = N_t0 # Copy the initial population into every run
    I[:, 0, 4] = 1000 # Seed 1,000 initial infections in age group 4
    S[:, 0, 4] -= I[:, 0, 4]

    # ---- Run Stochastic Model ----
    for t in tqdm.tqdm(
        iterable=range(1, t_max),
        desc="Generating SIRV Training Data",
        unit="days",
        disable=False,
        leave=True,
        bar_format=PROGRESS_BAR_FORMAT
    ):
        # State at the previous time step
        S_previous = S[:, t - 1, :]
        I_previous = I[:, t - 1, :]
        R_previous = R[:, t - 1, :]
        V_previous = V[:, t - 1, :]

        # Total population in each run and age group
        N_previous = (S_previous + I_previous + R_previous + V_previous)

        # Proportion infectious in every run and source age group
        infectious_prevalence = np.divide(
            I_previous, N_previous,
            out=np.zeros_like(I_previous, dtype=float),
            where=N_previous > 0
        )

        infection_rate = infectious_prevalence @ betas.T

        # Sample births and competing compartment transitions
        new_births = rng.poisson(lam=alpha * S_previous)
        S_routes = sample_routes(
            S_previous,
            np.stack([
                infection_rate,
                np.broadcast_to(rho, S_previous.shape),
                np.broadcast_to(delta, S_previous.shape),
                np.broadcast_to(theta, S_previous.shape)
            ], axis=-1)
        ) # S -> I, V, R, next age group, or S

        I_routes = sample_routes(
            I_previous,
            np.stack([
                np.broadcast_to(tau, I_previous.shape),
                np.broadcast_to(gamma, I_previous.shape),
                np.broadcast_to(delta, I_previous.shape),
                np.broadcast_to(theta, I_previous.shape)
            ], axis=-1)
        ) # I -> S, R through gamma, R through delta, next age group, or I

        R_routes = sample_routes(
            R_previous,
            np.broadcast_to(theta, R_previous.shape)[..., None]
        ) # R -> next age group or R

        V_routes = sample_routes(
            V_previous,
            np.stack([
                np.broadcast_to(kappa, V_previous.shape),
                np.broadcast_to(delta, V_previous.shape),
                np.broadcast_to(theta, V_previous.shape)
            ], axis=-1)
        ) # V -> S, R, next age group, or V

        # Unpack sampled routes
        new_infections, new_vaccinations, removed_from_S, aged_from_S, remaining_S = np.moveaxis(S_routes, -1, 0)
        recovered_to_S, recovered_to_R, removed_from_I, aged_from_I, remaining_I = np.moveaxis(I_routes, -1, 0)
        aged_from_R, remaining_R = np.moveaxis(R_routes, -1, 0)
        vaccine_waning, removed_from_V, aged_from_V, remaining_V = np.moveaxis(V_routes, -1, 0)

        # Update compartments
        S[:, t, :] = remaining_S + new_births + recovered_to_S + vaccine_waning
        I[:, t, :] = remaining_I + new_infections
        R[:, t, :] = remaining_R + recovered_to_R + removed_from_S + removed_from_I + removed_from_V
        V[:, t, :] = remaining_V + new_vaccinations

        # Add aging inflows
        S[:, t, 1:] += aged_from_S[:, :-1]
        I[:, t, 1:] += aged_from_I[:, :-1]
        R[:, t, 1:] += aged_from_R[:, :-1]
        V[:, t, 1:] += aged_from_V[:, :-1]

    # ---- Plot Median and Central 95% Simulation Intervals ----
    if plot_run:
        time = np.arange(t_max)
        lower, median, upper = np.percentile(
            a=np.stack([S, I, R, V]),
            q=[2.5, 50.0, 97.5],
            axis=1
        )

        fig, ax = plt.subplots(figsize=(9, 3), facecolor="white")
        for compartment, compartment_name in enumerate(["S", "I", "R", "V"]):
            for age_group in range(num_age_groups):
                colour = compartment_colours[compartment]
                opacity = opacities[age_group]
                ax.plot(
                    time,
                    median[compartment, :, age_group],
                    color=colour,
                    alpha=opacity,
                    label=f"${compartment_name}_{age_group}$"
                )
                ax.fill_between(
                    time,
                    lower[compartment, :, age_group],
                    upper[compartment, :, age_group],
                    color=colour,
                    alpha=0.15 * opacity,
                    linewidth=0
                )
        
        ax.set_xlabel("Time, $t$ (Days)")
        ax.set_ylabel("Population Size, $N$")
        ax.legend(title="Compartments", loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=2)
        fig.tight_layout(rect=[0, 0, 1, 1])
        plt.show()

    return S, I, R, V


if __name__ == "__main__":
    print("\n# ---- Create_SIRV_Training_Data.py Start ---- #")

    # Uses the LaTeX Computer Modern Serif fonts for visualisations
    plt.rcParams.update({"text.usetex": True, "font.family": "serif"})

    # ---- Run ----
    wallinga_betas = [
        [1.393*(10**-7), 0.259*(10**-7), 0.146*(10**-7), 0.284*(10**-7), 0.130*(10**-7), 0.094*(10**-7)],
        [0.259*(10**-7), 2.261*(10**-7), 0.266*(10**-7), 0.287*(10**-7), 0.170*(10**-7), 0.095*(10**-7)],
        [0.146*(10**-7), 0.266*(10**-7), 1.847*(10**-7), 0.418*(10**-7), 0.309*(10**-7), 0.123*(10**-7)],
        [0.284*(10**-7), 0.287*(10**-7), 0.418*(10**-7), 0.623*(10**-7), 0.407*(10**-7), 0.207*(10**-7)],
        [0.130*(10**-7), 0.170*(10**-7), 0.309*(10**-7), 0.407*(10**-7), 0.504*(10**-7), 0.272*(10**-7)],
        [0.094*(10**-7), 0.095*(10**-7), 0.123*(10**-7), 0.207*(10**-7), 0.272*(10**-7), 0.447*(10**-7)]
    ]

    alpha = np.zeros(num_age_groups)
    gamma = np.zeros(num_age_groups)
    delta = np.array([0.001, 0.005, 0.007, 0.009, 0.011, 0.013])
    theta = np.array([
        1 / (365 * 5),
        1 / (365 * 7),
        1 / (365 * 7),
        1 / (365 * 20),
        1 / (365 * 20),
        0.0
    ])
    rho = np.zeros(num_age_groups)
    kappa = np.zeros(num_age_groups)
    tau = np.repeat(0.286, num_age_groups)
    S, I, R, V = SIRV(
        t_max=365, # Number of simulated days
        betas=wallinga_betas,
        alpha=alpha,
        gamma=gamma,
        delta=delta,
        theta=theta,
        rho=rho,
        kappa=kappa,
        tau=tau,
        plot_run=True
    )

    print("# ---- Program End. ---- #\n")