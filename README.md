# Optimal Vaccination Strategies For Influenza Viruses With Machine Learning

## Introduction
In this work, we aim to identify optimal strategies for vaccine allocation to prevent large-scale influenza epidemics as they emerge. 

## The Mathematical Model
We implement a classic population-based SIRV (Susceptible, Infected, Removed/Deceased, and Vaccinated) epidemiological model with three age groups `0`, `1`, and `2`. Age group `0` is defined by those in who are between 0 and 18 years of age, age group `1` are those between 18 and 65 years of age, and age group `2` are those over 65. Our model is defined for each age group $`i={0, 1, 2}`$:
```math
\begin{align}
\frac{dS_i}{dt} &= \alpha_{i} S_{i} - \delta_{i} S_{i} - \left(\frac{\beta_{i,i} S_{i} I_{i}}{N_i} + \frac{\beta_{i,i+1} S_{i} I_{i+1}}{N_{i+1}} + \frac{\beta_{i,i+2} S_{i} I_{i+2}}{N_{i+2}}\right) - \rho_{i} S_{i} + \kappa_{i} V_{i} - \theta_{i} S_{i}, \\
\frac{dI_i}{dt} &= -\delta_{i} I_{i} + \left(\frac{\beta_{i,i} S_{i} I_{i}}{N_i} + \frac{\beta_{i,i+1} S_{i} I_{i+1}}{N_{i+1}} + \frac{\beta_{i,i+2} S_{i} I_{i+2}}{N_{i+2}}\right) - \gamma_{i} I_i - \theta_{i} I_i, \\
\frac{dR_i}{dt} &= \delta_{i} (S_i + I_i + V_i) + \gamma_{i} I_i - \theta_{i} R_i, \\
\frac{dV_i}{dt} &= \rho_{i} S_i - \kappa_{i} V_i - \delta_{i} V_i - \theta_{i} V_i, \\
\end{align}
```
where $`\alpha_i`$ is the mean birth rate, $`\gamma_i`$ is the mean death rate from infection, $`\delta_i`$ is the mean death rate from natural causes, $`\theta_i`$ is the rate of aging, $`\rho_i`$ is the rate of vaccination, and $`\kappa_i`$ is the rate of the loss of vaccination protection (whether by vaccination wear-off or viral genetic mutation). All units are in $`1/\mathrm{days}`$ and the subscripts are for age group $`i`$. The values $`\beta_{i,j}`$ are the contact rates between age groups $`i`$ and $`j`$. If $`j=i`$, then this is the contact rate with members of age group $`i`$ and themselves. Additions for the subscripts in $`\beta_{i,j}`$ are implemented modulo 3 so not to repeat calculations for age groups. The values $`N_{i}`$ are the population sizes of age group $`i`$.

## Results

## Requirements
To install the requirements, please run
```
pip install numpy scipy matplotlib tqdm
```
in your terminal.


## Citation
To cite this work, please use our BibTeX citation:
```
@article{...,
    title   = {...},
    author  = {...},
    journal = {...},
    year    = {...}
}
```
