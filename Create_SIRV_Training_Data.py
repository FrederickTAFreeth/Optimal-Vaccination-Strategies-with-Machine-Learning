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
def SIRV(t: float, y: List[float], alpha: List[float], beta: Union[np.ndarray, List[List[float]]], gamma: List[float], delta: List[float], theta: List[float], rho: List[float], kappa: List[float]) -> List[float]:
    S0, I0, R0, V0, S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3, S4, I4, R4, V4, S5, I5, R5, V5 = y

    N0 = S0 + I0 + R0 + V0
    N1 = S1 + I1 + R1 + V1
    N2 = S2 + I2 + R2 + V2
    N3 = S3 + I3 + R3 + V3
    N4 = S4 + I4 + R4 + V4
    N5 = S5 + I5 + R5 + V5

    age_group_0_inf_force = (beta[0][0]*S0*I0/N0 + beta[0][1]*S0*I1/N1 + beta[0][2]*S0*I2/N2 + beta[0][3]*S0*I3/N3 + beta[0][4]*S0*I4/N4 + beta[0][5]*S0*I5/N5)
    age_group_1_inf_force = (beta[1][0]*S1*I0/N0 + beta[1][1]*S1*I1/N1 + beta[1][2]*S1*I2/N2 + beta[1][3]*S1*I3/N3 + beta[1][4]*S1*I4/N4 + beta[1][5]*S1*I5/N5)
    age_group_2_inf_force = (beta[2][0]*S2*I0/N0 + beta[2][1]*S2*I1/N1 + beta[2][2]*S2*I2/N2 + beta[2][3]*S2*I3/N3 + beta[2][4]*S2*I4/N4 + beta[2][5]*S2*I5/N5)
    age_group_3_inf_force = (beta[3][0]*S3*I0/N0 + beta[3][1]*S3*I1/N1 + beta[3][2]*S3*I2/N2 + beta[3][3]*S3*I3/N3 + beta[3][4]*S3*I4/N4 + beta[3][5]*S3*I5/N5)
    age_group_4_inf_force = (beta[4][0]*S4*I0/N0 + beta[4][1]*S4*I1/N1 + beta[4][2]*S4*I2/N2 + beta[4][3]*S4*I3/N3 + beta[4][4]*S4*I4/N4 + beta[4][5]*S4*I5/N5)
    age_group_5_inf_force = (beta[5][0]*S5*I0/N0 + beta[5][1]*S5*I1/N1 + beta[5][2]*S5*I2/N2 + beta[5][3]*S5*I3/N3 + beta[5][4]*S5*I4/N4 + beta[5][5]*S5*I5/N5)

    dS0dt = alpha[0]*S0 - delta[0]*S0 - age_group_0_inf_force - rho[0]*S0 + kappa[0]*V0 - theta[0]*S0
    dI0dt = -delta[0]*I0 + age_group_0_inf_force - gamma[0]*I0 - theta[0]*I0
    dR0dt = delta[0]*(S0 + I0 + V0) + gamma[0]*I0 - theta[0]*R0
    dV0dt = rho[0]*S0 - kappa[0]*V0 - delta[0]*V0 - theta[0]*V0

    dS1dt = alpha[1]*S1 - delta[1]*S1 - age_group_1_inf_force - rho[1]*S1 + kappa[1]*V1 + theta[0]*S0 - theta[1]*S1
    dI1dt = -delta[1]*I1 + age_group_1_inf_force - gamma[1]*I1 + theta[0]*I0 - theta[1]*I1
    dR1dt = delta[1]*(S1 + I1 + V1) + gamma[1]*I1 + theta[0]*R0 - theta[1]*R1
    dV1dt = rho[1]*S1 - kappa[1]*V1 - delta[1]*V1 + theta[0]*V0 - theta[1]*V1

    dS2dt = alpha[2]*S2 - delta[2]*S2 - age_group_2_inf_force - rho[2]*S2 + kappa[2]*V2 + theta[1]*S1 - theta[2]*S2
    dI2dt = -delta[2]*I2 + age_group_2_inf_force - gamma[2]*I2 + theta[1]*I1 - theta[2]*I2
    dR2dt = delta[2]*(S2 + I2 + V2) + gamma[2]*I2 + theta[1]*R1 - theta[2]*R2
    dV2dt = rho[2]*S2 - kappa[2]*V2 - delta[2]*V2 + theta[1]*V1 - theta[2]*V2

    dS3dt = alpha[3]*S3 - delta[3]*S3 - age_group_3_inf_force - rho[3]*S3 + kappa[3]*V3 + theta[2]*S2 - theta[3]*S3
    dI3dt = -delta[3]*I3 + age_group_3_inf_force - gamma[3]*I3 + theta[2]*I2 - theta[3]*I3
    dR3dt = delta[3]*(S3 + I3 + V3) + gamma[3]*I3 + theta[2]*R2 - theta[3]*R3
    dV3dt = rho[3]*S3 - kappa[3]*V3 - delta[3]*V3 + theta[2]*V2 - theta[3]*V3

    dS4dt = alpha[4]*S4 - delta[4]*S4 - age_group_4_inf_force - rho[4]*S4 + kappa[4]*V4 + theta[3]*S3 - theta[4]*S4
    dI4dt = -delta[4]*I4 + age_group_4_inf_force - gamma[4]*I4 + theta[3]*I3 - theta[4]*I4
    dR4dt = delta[4]*(S4 + I4 + V4) + gamma[4]*I4 + theta[3]*R3 - theta[4]*R4
    dV4dt = rho[4]*S4 - kappa[4]*V4 - delta[4]*V4 + theta[3]*V3 - theta[4]*V4

    dS5dt = alpha[5]*S5 - delta[5]*S5 - age_group_5_inf_force - rho[5]*S5 + kappa[5]*V5 + theta[4]*S4
    dI5dt = -delta[5]*I5 + age_group_5_inf_force - gamma[5]*I5 + theta[4]*I4
    dR5dt = delta[5]*(S5 + I5 + V5) + gamma[5]*I5 + theta[4]*R4
    dV5dt = rho[5]*S5 - kappa[5]*V5 - delta[5]*V5 + theta[4]*V4

    return [dS0dt, dI0dt, dR0dt, dV0dt, dS1dt, dI1dt, dR1dt, dV1dt, dS2dt, dI2dt, dR2dt, dV2dt, dS3dt, dI3dt, dR3dt, dV3dt, dS4dt, dI4dt, dR4dt, dV4dt, dS5dt, dI5dt, dR5dt, dV5dt]


# ---- Running the SIRV ODE system ----
def run_SIRV(beta: Union[np.ndarray, List[List[float]]], plot_run: bool=False) -> Tuple[List[float]]:
    N0, N1, N2, N3, N4, N5 = 1000, 2000, 3000, 2500, 1500, 750 # Initial age group population sizes
    I0_t0, R0_t0, V0_t0 = 1, 0, 0 # Initial compartment sizes of age group 0
    I1_t0, R1_t0, V1_t0 = 1, 0, 0 # Initial compartment sizes of age group 1
    I2_t0, R2_t0, V2_t0 = 1, 0, 0 # Initial compartment sizes of age group 2
    I3_t0, R3_t0, V3_t0 = 1, 0, 0 # Initial compartment sizes of age group 3
    I4_t0, R4_t0, V4_t0 = 1, 0, 0 # Initial compartment sizes of age group 4
    I5_t0, R5_t0, V5_t0 = 1, 0, 0 # Initial compartment sizes of age group 5
    S0_t0 = N0 - I0_t0 - R0_t0 - V0_t0
    S1_t0 = N1 - I1_t0 - R1_t0 - V1_t0
    S2_t0 = N2 - I2_t0 - R2_t0 - V2_t0
    S3_t0 = N3 - I3_t0 - R3_t0 - V3_t0
    S4_t0 = N4 - I4_t0 - R4_t0 - V4_t0
    S5_t0 = N5 - I5_t0 - R5_t0 - V5_t0
    alpha = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # Mean birth rate (in 1/days) for each age group.
    gamma = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] # Mean death rate from infection (in 1/days) for each age group.
    delta = [0.001, 0.005, 0.007, 0.009, 0.011, 0.013] # Mean death rate from natural causes (in 1/days) for each age group.
    theta = [1/(365 * 5), 1/(365 * (12 - 5)), 1/(365 * (19 - 12)), 1/(365 * (39 - 19)), 1/(365 * (59 - 39))] # Rate of aging in 1/days.
    rho = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # Vaccination rate (in 1/days)
    kappa = [0.04, 0.04, 0.04, 0.04, 0.04, 0.04] # Loss of vaccination protection (in 1/days)
    t = np.linspace(0, t_max, t_max) # A grid of time points (in days)

    # Initial conditions vector of each SIRV compartment for each age group
    y_t0 = [S0_t0, I0_t0, R0_t0, V0_t0, S1_t0, I1_t0, R1_t0, V1_t0, S2_t0, I2_t0, R2_t0, V2_t0, S3_t0, I3_t0, R3_t0, V3_t0, S4_t0, I4_t0, R4_t0, V4_t0, S5_t0, I5_t0, R5_t0, V5_t0]

    # ---- Integrate the SIRV equations over the time grid, t ----
    sol = scipy.integrate.solve_ivp(
        fun=SIRV, t_span=(t[0], t[-1]), y0=y_t0, args=(alpha, beta, gamma, delta, theta, rho, kappa),
        dense_output=True, method="RK45"
    )
    S0, I0, R0, V0, S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3, S4, I4, R4, V4, S5, I5, R5, V5 = sol.sol(t)

    # Plot the data of a single run - useful for debugging
    if plot_run:
        plt.figure(facecolor='w', figsize=(13, 7))
        plt.plot(t, S0, compartment_colours[0], alpha=opacities[0], lw=2, label=f'$S_0$')
        plt.plot(t, S1, compartment_colours[0], alpha=opacities[1], lw=2, label=f'$S_1$')
        plt.plot(t, S2, compartment_colours[0], alpha=opacities[2], lw=2, label=f'$S_2$')
        plt.plot(t, S3, compartment_colours[0], alpha=opacities[3], lw=2, label=f'$S_3$')
        plt.plot(t, S4, compartment_colours[0], alpha=opacities[4], lw=2, label=f'$S_4$')
        plt.plot(t, S5, compartment_colours[0], alpha=opacities[5], lw=2, label=f'$S_5$')

        plt.plot(t, I0, compartment_colours[1], alpha=opacities[0], lw=2, label=f'$I_0$')
        plt.plot(t, I1, compartment_colours[1], alpha=opacities[1], lw=2, label=f'$I_1$')
        plt.plot(t, I2, compartment_colours[1], alpha=opacities[2], lw=2, label=f'$I_2$')
        plt.plot(t, I3, compartment_colours[1], alpha=opacities[3], lw=2, label=f'$I_3$')
        plt.plot(t, I4, compartment_colours[1], alpha=opacities[4], lw=2, label=f'$I_4$')
        plt.plot(t, I5, compartment_colours[1], alpha=opacities[5], lw=2, label=f'$I_5$')

        plt.plot(t, R0, compartment_colours[2], alpha=opacities[0], lw=2, label=f'$R_0$')
        plt.plot(t, R1, compartment_colours[2], alpha=opacities[1], lw=2, label=f'$R_1$')
        plt.plot(t, R2, compartment_colours[2], alpha=opacities[2], lw=2, label=f'$R_2$')
        plt.plot(t, R3, compartment_colours[2], alpha=opacities[3], lw=2, label=f'$R_3$')
        plt.plot(t, R4, compartment_colours[2], alpha=opacities[4], lw=2, label=f'$R_4$')
        plt.plot(t, R5, compartment_colours[2], alpha=opacities[5], lw=2, label=f'$R_5$')

        plt.plot(t, V0, compartment_colours[3], alpha=opacities[0], lw=2, label=f'$V_0$')
        plt.plot(t, V1, compartment_colours[3], alpha=opacities[1], lw=2, label=f'$V_1$')
        plt.plot(t, V2, compartment_colours[3], alpha=opacities[2], lw=2, label=f'$V_2$')
        plt.plot(t, V3, compartment_colours[3], alpha=opacities[3], lw=2, label=f'$V_3$')
        plt.plot(t, V4, compartment_colours[3], alpha=opacities[4], lw=2, label=f'$V_4$')
        plt.plot(t, V5, compartment_colours[3], alpha=opacities[5], lw=2, label=f'$V_5$')

        plt.plot(t, (S0 + I0 + R0 + V0), 'orange', alpha=opacities[0], lw=2, label=f'$A_0$')
        plt.plot(t, (S1 + I1 + R1 + V1), 'orange', alpha=opacities[1], lw=2, label=f'$A_1$')
        plt.plot(t, (S2 + I2 + R2 + V2), 'orange', alpha=opacities[2], lw=2, label=f'$A_2$')
        plt.plot(t, (S3 + I3 + R3 + V3), 'orange', alpha=opacities[3], lw=2, label=f'$A_3$')
        plt.plot(t, (S4 + I4 + R4 + V4), 'orange', alpha=opacities[4], lw=2, label=f'$A_4$')
        plt.plot(t, (S5 + I5 + R5 + V5), 'orange', alpha=opacities[5], lw=2, label=f'$A_5$')

        plt.xlabel('Time (Days)')
        plt.ylabel('Population Size')
        plt.tight_layout(rect=[0, 0, 0.875, 1])
        plt.legend(title="Compartments", loc="center left", bbox_to_anchor=(1, 0.5))
        plt.show()
    return t, S0, I0, R0, V0, S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3, S4, I4, R4, V4, S5, I5, R5, V5


# ---- Global Parameters ----
NUM_CPU_CORES = os.cpu_count() # Number of CPU cores
NUM_RUNS = 64 # Number of times to run the model
t_max = 128 # Final integration time (days)
compartment_colours = ["green", "red", "black", "blue"] # Compartment colours for plotting
opacities = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1] # Opacities for age groups for plotting
np.random.seed(314159)


if __name__ == "__main__":
    print("# ---- Create_SIRV_Training_Data.py Start ---- #")

    # Uses the LaTeX Computer Modern Serif fonts for visualisations
    plt.rcParams.update({"text.usetex": True, "font.family": "serif"})

    # ---- Single Run ----
    print("Calculating a single run... ", end="")
    example_beta = [
        [0.23, 0.45, 0.67, 0.12, 0.89, 0.34],
        [0.45, 0.78, 0.91, 0.11, 0.22, 0.33],
        [0.67, 0.91, 0.66, 0.77, 0.88, 0.99],
        [0.12, 0.11, 0.77, 0.45, 0.56, 0.67],
        [0.89, 0.22, 0.88, 0.56, 0.34, 0.56],
        [0.34, 0.33, 0.99, 0.67, 0.56, 0.23]
    ]
    # Set plot_run to True to see a plot of the epideomiological system.
    t, S0, I0, R0, V0, S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3, S4, I4, R4, V4, S5, I5, R5, V5 = run_SIRV(beta=example_beta, plot_run=True)
    print("Done.")

    # ---- Model Parameters For Multithreaded Runs ----
    print(f"Running the model for {NUM_RUNS} runs... ", end="")
    # Using ProcessPoolExecutor with the number of CPU cores as max_workers
    SIRV_Model_Data = {
        # Each value in this dictionary will eventually have shape (NUM_RUNS, t_max)
        "S_0": np.empty((0, t_max)), "I_0": np.empty((0, t_max)), "R_0": np.empty((0, t_max)), "V_0": np.empty((0, t_max)),
        "S_1": np.empty((0, t_max)), "I_1": np.empty((0, t_max)), "R_1": np.empty((0, t_max)), "V_1": np.empty((0, t_max)),
        "S_2": np.empty((0, t_max)), "I_2": np.empty((0, t_max)), "R_2": np.empty((0, t_max)), "V_2": np.empty((0, t_max)),
        "S_3": np.empty((0, t_max)), "I_3": np.empty((0, t_max)), "R_3": np.empty((0, t_max)), "V_3": np.empty((0, t_max)),
        "S_4": np.empty((0, t_max)), "I_4": np.empty((0, t_max)), "R_4": np.empty((0, t_max)), "V_4": np.empty((0, t_max)),
        "S_5": np.empty((0, t_max)), "I_5": np.empty((0, t_max)), "R_5": np.empty((0, t_max)), "V_5": np.empty((0, t_max))
    }
    min_beta, max_beta = 0, 1
    # This is a multidimentional version of the example beta matrix above.
    betas = min_beta + (np.random.random(size=(NUM_RUNS, 6, 6)) * (max_beta - min_beta))
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
    plt.figure(facecolor='w', figsize=(13, 7))
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