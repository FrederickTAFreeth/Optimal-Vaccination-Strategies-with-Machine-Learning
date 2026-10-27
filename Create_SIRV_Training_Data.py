"""
Optimal Vaccination Strategies with Machine Learning

Authors:
    Frederick T. A. Freeth (https://orcid.org/0009-0007-9893-2250),
    Brian Gardener (https://orcid.org/0000-0001-6199-8917),
    Joaquin M. Prada (https://orcid.org/0000-0003-4699-5931)

BibTex Reference:
{
    ...
}
"""

# ---- Importing Libraries ----
import numpy as np
import tqdm
import matplotlib.pyplot as plt


# ---- Global Parameters ----
num_age_groups = 6 # Number of age groups

def SIRV(
    *,
    t_max: int=64,
    N_t0,
    I_t0,
    betas,
    alpha,
    gamma,
    delta,
    mu,
    theta,
    rho,
    kappa,
    tau,
    plot_run=False,
    rng_val: int=314159,
    num_runs: int=2048
):
    """
    Simulate a batch of stochastic age-structured SIRVD epidemics.

    Array shapes:
        S, I, R, V, D, D_infection: (num_runs, t_max, num_age_groups)
        B: (num_runs, t_max)
        betas: (num_age_groups, num_age_groups)
        Other model parameters:(num_age_groups,)

    Parameter meanings:
        t_max: Number of stored daily observations, including the initial state.
            The model therefore performs t_max - 1 daily transitions.
        alpha: Per-capita contribution to the total population birth rate.
            Births are summed over the living population and all newborns
            enter S in age group 0.
        gamma: Recovery rate from I to R.
        delta: Background mortality rate from S, I, R, and V to D.
        mu: Mortality rate caused by infection.
        theta: Aging rate into the next age group.
        rho: Vaccination rate from S to V.
        kappa: Loss of vaccine protection from V to S.
        tau: Loss of natural immunity from R to S.

    Returns:
        Dictionary containing S, I, R, and V compartment trajectories,
        cumulative deaths D, cumulative infection-attributable deaths
        D_infection, and cumulative births B.
    """
    rng = np.random.default_rng(rng_val)
    N_t0 = np.asarray(N_t0, dtype=np.int32)
    betas = np.asarray(betas, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    delta = np.asarray(delta, dtype=float)
    mu = np.asarray(mu, dtype=float)
    theta = np.asarray(theta, dtype=float)
    rho = np.asarray(rho, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    tau = np.asarray(tau, dtype=float)

    # ---- Check Parameter Shapes ----
    if N_t0.shape != (num_age_groups,):
        raise ValueError(f"N_t0 must have shape ({num_age_groups},)")

    if betas.shape != (num_age_groups, num_age_groups):
        raise ValueError(f"betas must have shape ({num_age_groups}, {num_age_groups})")

    if not np.all(np.isfinite(betas)):
        raise ValueError("betas must contain finite transmission rates")

    for parameter, parameter_name in [
        (alpha, "alpha"),
        (gamma, "gamma"),
        (delta, "delta"),
        (mu, "mu"),
        (theta, "theta"),
        (rho, "rho"),
        (kappa, "kappa"),
        (tau, "tau")
    ]:
        if parameter.shape != (num_age_groups,):
            raise ValueError(f"{parameter_name} must have shape ({num_age_groups},)")

        if np.any(parameter < 0):
            raise ValueError(f"{parameter_name} cannot contain negative rates")

        if not np.all(np.isfinite(parameter)):
            raise ValueError(f"{parameter_name} must contain finite rates")

    if np.any(N_t0 < 0):
        raise ValueError("N_t0 cannot contain negative populations")

    if I_t0 < 0 or I_t0 > N_t0[4]:
        raise ValueError("I_t0 must be between zero and the initial population of age group 4")

    if not np.isclose(theta[-1], 0.0):
        raise ValueError("The final age group must have an aging rate of zero")

    if np.any(betas < 0):
        raise ValueError("betas cannot contain negative transmission rates")

    if t_max < 1:
        raise ValueError("t_max must be at least one")

    if num_runs < 1:
        raise ValueError("num_runs must be at least one")

    if not isinstance(I_t0, (int, np.integer)):
        raise TypeError("I_t0 must be an integer")

    def sample_routes(n, rates):
        """
        Sample competing transitions and a final remain-in-place route.
        """
        total_rate = rates.sum(axis=-1, keepdims=True)
        p = np.divide(
            rates,
            total_rate,
            out=np.zeros_like(rates),
            where=total_rate > 0
        ) * (-np.expm1(-total_rate))
        p = np.concatenate([p, np.exp(-total_rate)], axis=-1)
        return rng.multinomial(n=n, pvals=p)

    # ---- Initialise State Arrays ----
    shape = (num_runs, t_max, num_age_groups)
    S = np.zeros(shape, dtype=np.int32)
    I = np.zeros(shape, dtype=np.int32)
    R = np.zeros(shape, dtype=np.int32)
    V = np.zeros(shape, dtype=np.int32)
    D = np.zeros(shape, dtype=np.int32) # Cumulative deaths by age at death
    D_infection = np.zeros(shape, dtype=np.int32)
    B = np.zeros((num_runs, t_max), dtype=np.int32) # Cumulative births
    S[:, 0, :] = N_t0

    # Seed infection in age group 4
    I[:, 0, 4] = I_t0
    S[:, 0, 4] -= I[:, 0, 4]

    # ---- Run Stochastic Model ----
    for t in tqdm.tqdm(
        iterable=range(1, t_max),
        desc="Generating SIRVD Training Data",
        unit="days",
        disable=False,
        leave=True,
        bar_format=(
            "{desc}: {n_fmt}/{total_fmt} {unit} "
            "({percentage:3.1f}%) | Elapsed: {elapsed} | "
            "Remaining: {remaining} | {rate_fmt}"
        )
    ):
        # State at the previous time step
        S_previous = S[:, t - 1, :]
        I_previous = I[:, t - 1, :]
        R_previous = R[:, t - 1, :]
        V_previous = V[:, t - 1, :]
        D_previous = D[:, t - 1, :]

        # Living population in each run and age group
        N_previous = (S_previous + I_previous + R_previous + V_previous)

        # Per-susceptible infection hazard generated by infectious
        # individuals in every source age group
        infection_rate = I_previous @ betas.T

        # Alpha represents a per-capita contribution to the total birth
        # rate. Births are summed across the living population and all
        # newborns enter susceptible age group 0.
        new_births = rng.poisson(lam=np.sum(alpha * N_previous, axis=1))
        B[:, t] = B[:, t - 1] + new_births

        # Susceptible Routes: S -> I, V, D, next age group, remain S
        S_routes = sample_routes(
            n=S_previous,
            rates=np.stack(
                arrays=[
                    infection_rate,
                    np.broadcast_to(rho, S_previous.shape),
                    np.broadcast_to(delta, S_previous.shape),
                    np.broadcast_to(theta, S_previous.shape)
                ],
                axis=-1
            )
        )

        # Infected Routes: I -> R, D from infection, D from background
        # mortality, next age group, or remain in I
        I_routes = sample_routes(
            n=I_previous,
            rates=np.stack(
                arrays=[
                    np.broadcast_to(gamma, I_previous.shape),
                    np.broadcast_to(mu, I_previous.shape),
                    np.broadcast_to(delta, I_previous.shape),
                    np.broadcast_to(theta, I_previous.shape)
                ],
                axis=-1
            )
        )

        # Recovered Routes: R -> S, D, next age group, or remain in R
        R_routes = sample_routes(
            n=R_previous,
            rates=np.stack(
                arrays=[
                    np.broadcast_to(tau, R_previous.shape),
                    np.broadcast_to(delta, R_previous.shape),
                    np.broadcast_to(theta, R_previous.shape)
                ],
                axis=-1
            )
        )

        # Vaccinated Routes: V -> S, D, next age group, or remain in V
        V_routes = sample_routes(
            n=V_previous,
            rates=np.stack(
                arrays=[
                    np.broadcast_to(kappa, V_previous.shape),
                    np.broadcast_to(delta, V_previous.shape),
                    np.broadcast_to(theta, V_previous.shape)
                ],
                axis=-1
            )
        )

        # ---- Unpack Sampled Routes ----
        new_infections, new_vaccinations, deaths_from_S, aged_from_S, remaining_S = np.moveaxis(S_routes, -1, 0)
        new_recoveries, deaths_from_infection, deaths_from_I, aged_from_I, remaining_I = np.moveaxis(I_routes, -1, 0)
        immunity_waning, deaths_from_R, aged_from_R, remaining_R = np.moveaxis(R_routes, -1, 0)
        vaccine_waning, deaths_from_V, aged_from_V, remaining_V = np.moveaxis(V_routes, -1, 0)

        # ---- Update Living Compartments ----
        S[:, t, :] = (remaining_S + immunity_waning + vaccine_waning)
        I[:, t, :] = (remaining_I + new_infections)
        R[:, t, :] = (remaining_R + new_recoveries)
        V[:, t, :] = (remaining_V + new_vaccinations)

        # All newborns enter susceptible age group 0
        S[:, t, 0] += new_births

        # ---- Add Aging Inflows ----
        S[:, t, 1:] += aged_from_S[:, :-1]
        I[:, t, 1:] += aged_from_I[:, :-1]
        R[:, t, 1:] += aged_from_R[:, :-1]
        V[:, t, 1:] += aged_from_V[:, :-1]

        # Deaths remain indexed by age at death and do not age.
        D[:, t, :] = (D_previous + deaths_from_S + deaths_from_I + deaths_from_R + deaths_from_V + deaths_from_infection)
        D_infection[:, t, :] = (D_infection[:, t - 1, :] + deaths_from_infection)

    # ---- Validate Population Accounting ----
    accounted_population = (S.sum(axis=2) + I.sum(axis=2) + R.sum(axis=2) + V.sum(axis=2) + D.sum(axis=2))
    expected_population = (N_t0.sum() + B)
    if not np.array_equal(accounted_population, expected_population):
        difference = (accounted_population - expected_population)
        run_idx, time_idx = np.unravel_index(np.argmax(np.abs(difference)), difference.shape)
        raise RuntimeError(
            "Population accounting failed. "
            f"Difference: {difference[run_idx, time_idx]} "
            f"at run {run_idx}, time index {time_idx}"
        )

    if np.any(np.diff(B, axis=1) < 0):
        raise RuntimeError("Cumulative births decreased")

    if np.any(np.diff(D, axis=1) < 0):
        raise RuntimeError("Cumulative deaths decreased")

    if np.any(np.diff(D_infection, axis=1) < 0):
        raise RuntimeError("Cumulative infection-attributable deaths decreased")

    if np.any(D_infection > D):
        raise RuntimeError("Infection-attributable deaths exceed total deaths")

    if any(np.any(compartment < 0) for compartment in [S, I, R, V, D, D_infection, B]):
        raise RuntimeError("Negative compartment count generated")

    # ---- Plot Median and Central 95% Simulation Intervals ----
    if plot_run:
        time = np.arange(t_max)
        compartment_data = [S, I, R, V, D]
        quantiles = [
            np.percentile(compartment, q=[2.5, 50.0, 97.5], axis=0)
            for compartment in compartment_data
        ]
        compartment_names = ["S", "I", "R", "V", "D"]
        compartment_colours = ["green", "red", "orange", "blue", "black"]
        opacities = np.linspace(start=0.1, stop=1.0, num=num_age_groups)
        fig, ax = plt.subplots(figsize=(9, 4), facecolor="white")
        for compartment, compartment_name in enumerate(compartment_names):
            lower, median, upper = quantiles[compartment]
            for age_group in range(num_age_groups):
                colour = compartment_colours[compartment]
                opacity = opacities[age_group]
                ax.plot(
                    time,
                    median[:, age_group],
                    color=colour,
                    alpha=opacity,
                    label=f"${compartment_name}_{age_group}$"
                )
                ax.fill_between(
                    time,
                    lower[:, age_group],
                    upper[:, age_group],
                    color=colour,
                    alpha=0.15 * opacity,
                    linewidth=0
                )

        ax.set_xlabel("Time, $t$ (Days)")
        ax.set_ylabel("Population Size at Time $t$, $N(t)$")
        ax.legend(
            title="Compartments",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            ncol=2
        )
        fig.tight_layout()
        plt.show()

    return {"S": S, "I": I, "R": R, "V": V, "D": D, "D_infection": D_infection, "B": B}

if __name__ == "__main__":
    print("\n# ---- Create_SIRV_Training_Data.py Start ---- #")

    # Uses the LaTeX Computer Modern Serif fonts for visualisations
    plt.rcParams.update({"text.usetex": True, "font.family": "serif"})

    # Approximate fatality probability conditional on clinical infection:
    # Dutch mortality: approximately 1 death per 3,700 inhabitants.
    # https://pmc.ncbi.nlm.nih.gov/articles/PMC2555330/. 1957 clinical attack
    # rate: approximately 31%. https://www.jstor.org/stable/25385782.
    infectious_period = 2.6
    recovery_rate = np.repeat(1 / infectious_period, num_age_groups)
    background_mortality_rate = np.repeat(7.55 / 1000 / 365, num_age_groups)
    aging_rate = 1 / np.array([365 * 6, 365 * 7, 365 * 7, 365 * 20, 365 * 20, np.inf])
    influenza_fatality_probability = ((1 / 3700) / 0.31)
    infection_mortality_rate = (
        influenza_fatality_probability * (
            recovery_rate + background_mortality_rate + aging_rate
        )/ (1 - influenza_fatality_probability)
    )

    # ---- Run ----
    epi = SIRV(
        t_max=128, # 128 daily observations, including the initial state
        N_t0 = np.array([1060, 1265, 1642, 4857, 3312, 2477], dtype=int) * 10**3,
        I_t0=1000,
        betas=np.array([
            # https://www.pnas.org/doi/full/10.1073/pnas.0908491107
            [1.393e-7, 0.259e-7, 0.146e-7, 0.284e-7, 0.130e-7, 0.094e-7],
            [0.259e-7, 2.261e-7, 0.266e-7, 0.287e-7, 0.170e-7, 0.095e-7],
            [0.146e-7, 0.266e-7, 1.847e-7, 0.418e-7, 0.309e-7, 0.123e-7],
            [0.284e-7, 0.287e-7, 0.418e-7, 0.623e-7, 0.407e-7, 0.207e-7],
            [0.130e-7, 0.170e-7, 0.309e-7, 0.407e-7, 0.504e-7, 0.272e-7],
            [0.094e-7, 0.095e-7, 0.123e-7, 0.207e-7, 0.272e-7, 0.447e-7]
        ]),

        # https://www.cbs.nl/en-gb/figures/detail/37422eng
        alpha=np.repeat(21.5 / 1000 / 365, num_age_groups),

        # https://bio-protocol.org/exchange/minidetail?id=8745685&type=30
        gamma=recovery_rate,

        # Dutch crude mortality was 7.5 per 1,000 in 1950 and
        # 7.6 per 1,000 in 1960. https://www.cbs.nl/en-gb/figures/detail/37979eng.
        # Temporary uniform background mortality. This should eventually
        # be replaced with age-specific Dutch life-table hazards.
        delta=background_mortality_rate,

        # Convert infection-fatality probability to a competing mortality
        # hazard relative to recovery.
        mu=infection_mortality_rate,

        # Age-group widths: 0-5, 6-12, 13-19, 20-39, 40-59, and 60+ years. Aging
        # is represented by a constant hazard equal to the inverse age-band width.
        theta=aging_rate,

        # No vaccination coverage
        rho=np.zeros(num_age_groups),
        
        # Influenza vaccine antibody responses commonly decline within
        # approximately six months, although protection can persist across
        # a typical influenza season https://pubmed.ncbi.nlm.nih.gov/37331840/.
        kappa=np.repeat(1 / 180, num_age_groups),
        
        # Protective antibodies following pandemic influenza infection
        # persisted at high levels for at least approximately 15 months.
        # https://pmc.ncbi.nlm.nih.gov/articles/PMC4351579/
        tau=np.repeat(1 / (365 * 1.25), num_age_groups),

        # Plot the results
        plot_run=True
    )

    print("# ---- Program End. ---- #\n")