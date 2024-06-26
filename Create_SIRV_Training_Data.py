# #!/usr/bin/env python3
# ---- Importing Libraries ----
import os
import numpy as np
import tqdm
import json
import scipy
import matplotlib.pyplot as plt
import concurrent.futures
from typing import Union, List, Tuple

# To install requirements:
# pip install numpy scipy matplotlib tqdm

# ---- The SIRV model differential equations ----
def SIRV(t: float, y: List[float], alpha: List[float], beta: Union[np.ndarray,  List[List[float]]], gamma: List[float], delta: List[float], theta: List[float], rho: List[float], kappa: List[float]) -> List[float]:
    S0, I0, R0, V0, S1, I1, R1, V1, S2, I2, R2, V2 = y

    N0 = S0 + I0 + R0 + V0
    N1 = S1 + I1 + R1 + V1
    N2 = S2 + I2 + R2 + V2

    nu_0 = np.random.beta(a=2, b=5, size=None) # Stochastic infectious term
    nu_1 = np.random.beta(a=2, b=2, size=None) # Stochastic infectious term
    nu_2 = np.random.beta(a=5, b=1, size=None) # Stochastic infectious term

    age_group_0_inf_force = (beta[0][0]*S0*I0/N0 + beta[0][1]*S0*I1/N1 + beta[0][2]*S0*I2/N2)*nu_0
    age_group_1_inf_force = (beta[1][0]*S1*I0/N0 + beta[1][1]*S1*I1/N1 + beta[1][2]*S1*I2/N2)*nu_1
    age_group_2_inf_force = (beta[2][0]*S2*I0/N0 + beta[2][1]*S2*I1/N1 + beta[2][2]*S2*I2/N2)*nu_2

    dS0dt = alpha[0]*S0 - delta[0]*S0 - age_group_0_inf_force - rho[0]*S0 + kappa[0]*V0 - theta[0]*S0
    dI0dt = -delta[0]*I0 + age_group_0_inf_force - gamma[0]*I0 - theta[0]*I0
    dR0dt = delta[0]*(S0 + I0 + V0) + gamma[0]*I0 - theta[0]*R0
    dV0dt = rho[0]*S0 - kappa[0]*V0 - delta[0]*V0 - theta[0]*V0

    dS1dt = alpha[1]*S1 - delta[1]*S1 - age_group_1_inf_force - rho[1]*S1 + kappa[1]*V1 + theta[0]*S0 - theta[1]*S1
    dI1dt = -delta[1]*I1 + age_group_1_inf_force - gamma[1]*I1 + theta[0]*I0 - theta[1]*I1
    dR1dt = delta[1]*(S1 + I1 + V1) + gamma[1]*I1 + theta[0]*R0 - theta[1]*R1
    dV1dt = rho[1]*S1 - kappa[1]*V1 - delta[1]*V1 + theta[0]*V0 - theta[1]*V1

    dS2dt = alpha[2]*S2 - delta[2]*S2 - age_group_2_inf_force - rho[2]*S2 + kappa[2]*V2 + theta[1]*S1
    dI2dt = -delta[2]*I2 + age_group_2_inf_force - gamma[2]*I2 + theta[1]*I1
    dR2dt = delta[2]*(S2 + I2 + V2) + gamma[2]*I2 + theta[1]*R1
    dV2dt = rho[2]*S2 - kappa[2]*V2 - delta[2]*V2 + theta[1]*V1
    return [dS0dt, dI0dt, dR0dt, dV0dt, dS1dt, dI1dt, dR1dt, dV1dt, dS2dt, dI2dt, dR2dt, dV2dt]


# ---- Running the SIRV ODE system ----
def run_SIRV(beta: Union[np.ndarray,  List[List[float]]], plot_run: bool=False) -> Tuple[List[float]]:
    N0, N1, N2 = 3000, 6000, 1000 # Inital age group population sizes
    I0_t0, R0_t0, V0_t0 = 1, 0, 0 # Initial compartment sizes of age group 0
    I1_t0, R1_t0, V1_t0 = 1, 0, 0 # Initial compartment sizes of age group 1
    I2_t0, R2_t0, V2_t0 = 1, 0, 0 # Initial compartment sizes of age group 2
    S0_t0 = N0 - I0_t0 - R0_t0 - V0_t0
    S1_t0 = N1 - I1_t0 - R1_t0 - V1_t0
    S2_t0 = N2 - I2_t0 - R2_t0 - V2_t0
    alpha = [0.0, 0.0, 0.0] # Mean birth rate (in 1/days) for each age group.
    gamma = [0.1, 0.2, 0.3] # Mean death rate from infection (in 1/days) for each age group.
    delta = [0.001, 0.005, 0.007] # Mean death rate from natural causes (in 1/days) for each age group.
    theta = [1/(365 * 18), 1/(365 * (65 - 18))] # Rate of aging in 1/days. Age groups are under 18s, 18-65s and over 65s
    rho = [0.0, 0.0, 0.0] # Vaccination rate (in 1/days)
    kappa = [0.04, 0.04, 0.04] # Loss of vaccination protection (in 1/days)
    t = np.linspace(0, t_max, t_max) # A grid of time points (in days)

    # Initial conditions vector of each SIRV compartment for each age group
    y_t0 = S0_t0, I0_t0, R0_t0, V0_t0, S1_t0, I1_t0, R1_t0, V1_t0, S2_t0, I2_t0, R2_t0, V2_t0

    # ---- Integrate the SIRV equations over the time grid, t ----
    S0, I0, R0, V0, S1, I1, R1, V1, S2, I2, R2, V2 = scipy.integrate.solve_ivp(
        fun=SIRV, t_span=(t[0], t[-1]), y0=y_t0, args=(alpha, beta, gamma, delta, theta, rho, kappa),
        dense_output=True, method="RK45"
    ).sol(t)

    # Plot the data of a single run - useful for debugging
    if plot_run:
        plt.figure(facecolor='w', figsize = (9, 4))
        plt.plot(t, S0, compartment_colours[0], alpha=opacities[0], lw=2, label=f'$S_0$')
        plt.plot(t, S1, compartment_colours[0], alpha=opacities[1], lw=2, label=f'$S_1$')
        plt.plot(t, S2, compartment_colours[0], alpha=opacities[2], lw=2, label=f'$S_2$')
        plt.plot(t, I0, compartment_colours[1], alpha=opacities[0], lw=2, label=f'$I_0$')
        plt.plot(t, I1, compartment_colours[1], alpha=opacities[1], lw=2, label=f'$I_1$')
        plt.plot(t, I2, compartment_colours[1], alpha=opacities[2], lw=2, label=f'$I_2$')
        plt.plot(t, R0, compartment_colours[2], alpha=opacities[0], lw=2, label=f'$R_0$')
        plt.plot(t, R1, compartment_colours[2], alpha=opacities[1], lw=2, label=f'$R_1$')
        plt.plot(t, R2, compartment_colours[2], alpha=opacities[2], lw=2, label=f'$R_2$')
        plt.plot(t, V0, compartment_colours[3], alpha=opacities[0], lw=2, label=f'$V_0$')
        plt.plot(t, V1, compartment_colours[3], alpha=opacities[1], lw=2, label=f'$V_1$')
        plt.plot(t, V2, compartment_colours[3], alpha=opacities[2], lw=2, label=f'$V_2$')
        plt.plot(t, (S0 + I0 + R0 + V0), 'orange', alpha=opacities[0], lw=2, label=f'$A_0$')
        plt.plot(t, (S1 + I1 + R1 + V1), 'orange', alpha=opacities[1], lw=2, label=f'$A_1$')
        plt.plot(t, (S2 + I2 + R2 + V2), 'orange', alpha=opacities[2], lw=2, label=f'$A_2$')
        plt.xlabel('Time (Days)')
        plt.ylabel('Population Size')
        plt.tight_layout(rect=[0, 0, 0.875, 1])
        plt.legend(title="Compartments", loc="center left", bbox_to_anchor=(1, 0.5))
        plt.show()
    return t, S0, I0, R0, V0, S1, I1, R1, V1, S2, I2, R2, V2


# ---- Global Parameters ----
NUM_CPU_CORES = os.cpu_count() # Number of CPU cores
NUM_RUNS = 64 # Number of times to run the model
t_max = 128 # Final integration time (days)
compartment_colours = ["green", "red", "black", "blue"] # Compartment colours for plotting
opacities = [0.8, 0.5, 0.2] # Opacities for age groups for plotting
np.random.seed(314159)


if __name__ == "__main__":
    print("# ---- Create_SIRV_Training_Data.py Start ---- #")

    # Uses the LaTeX Computer Modern Serif fonts for visualisations
    plt.rcParams.update({"text.usetex": True, "font.family": "serif"})

    # ---- Single Run ----
    print("Calculating a single run... ", end="")
    example_beta = [
        [0.6, 0.5, 0.2],
        [0.5, 0.4, 0.4],
        [0.2, 0.4, 0.7]
    ]
    # Set plot_run to True to see a plot of the epideomiological system.
    t, S0, I0, R0, V0, S1, I1, R1, V1, S2, I2, R2, V2 = run_SIRV(beta=example_beta, plot_run=True)
    print("Done.")


    # ---- Model Parameters For Multithreaded Runs ----
    print(f"Running the model for {NUM_RUNS} runs... ", end="")
    # Using ProcessPoolExecutor with the number of CPU cores as max_workers
    SIRV_Model_Data = {
        # Each value in this dictionary will eventually have shape (NUM_RUNS, t_max)
        "S_0": np.empty((0, t_max)), "I_0": np.empty((0, t_max)), "R_0": np.empty((0, t_max)), "V_0": np.empty((0, t_max)),
        "S_1": np.empty((0, t_max)), "I_1": np.empty((0, t_max)), "R_1": np.empty((0, t_max)), "V_1": np.empty((0, t_max)),
        "S_2": np.empty((0, t_max)), "I_2": np.empty((0, t_max)), "R_2": np.empty((0, t_max)), "V_2": np.empty((0, t_max))
    }
    min_beta, max_beta = 0, 1
    # This is a multidimentional version of the example beta matrix above.
    betas = min_beta + (np.random.random(size=(NUM_RUNS, 3, 3)) * (max_beta - min_beta))
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(NUM_CPU_CORES, NUM_RUNS)) as executor:
        futures = [executor.submit(run_SIRV, beta) for beta in betas]
        for i, f in enumerate(tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures))):
            # Unpack the results and sort them into a dictonary to dump into a JSON file
            results = f.result() # Tuple of t, S0, I0, R0, V0, S1, I1, R1, V1, S2, I2, R2, and V2 data
            
            # Vertically stack the new results onto the existing arrays in the dictionary
            for key, result in zip(SIRV_Model_Data.keys(), results[1:]):  # Skip the first result 't'.
                SIRV_Model_Data[key] = np.vstack((SIRV_Model_Data[key], result))
    print("Done.\n")


    # Write file to JSON as the data produced is very multidimensional
    print("Writing file to JSON... ", end="")
    with open("./SIRV_Model_Data.json", "w") as output_json:
        json.dump({key: value.tolist() for key, value in SIRV_Model_Data.items()}, output_json)
    print("Done.")

    # Read a pre-saved SIRV_Model_Data JSON dictionary
    print("Reading file from JSON... ", end="")
    with open("./SIRV_Model_Data.json", "r") as input_json:
        SIRV_Model_Data = json.load(input_json)
        SIRV_Model_Data = {key: np.array(value) for key, value in SIRV_Model_Data.items()}
    print("Done.")


    # ---- Plot the compartment sizes across all runs with a 95% Credible Interval ----
    print("Plotting multi-run results... ", end="")
    plt.figure(facecolor='w', figsize = (9, 4))
    t_grid = np.arange(start=0, stop=t_max, step=1)
    for i, (compartment_name, compartment_data) in enumerate(SIRV_Model_Data.items()):
        alpha_tmp = np.repeat(opacities, len(compartment_colours))[i]
        colour_tmp = compartment_colours[i % len(compartment_colours)]
        CIs_low = np.percentile(compartment_data, 2.5, axis=0)
        CIs_high = np.percentile(compartment_data, 97.5, axis=0)
        means = np.mean(compartment_data, axis=0)
        plt.plot(t_grid, means, label=f"${compartment_name}$", alpha=alpha_tmp, color=colour_tmp)
        plt.fill_between(t_grid, CIs_low, CIs_high, color=colour_tmp, alpha=alpha_tmp)

    plt.xlabel("Time (Days)")
    plt.ylabel(f"Population Sizes Across {NUM_RUNS} Runs")
    plt.tight_layout(rect=[0, 0, 0.875, 1])
    plt.legend(title="Compartments", loc="center left", bbox_to_anchor=(1, 0.5))
    plt.show()
    print("Done.")
    print("# ---- Program End. ---- #")