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
import os
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import lightning as L
import torch

from torch import nn
torch.set_float32_matmul_precision("high")
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import random_split

# To install depenancies:
# python -m pip install -U triton-windows
# pip install numpy matplotlib lightning tqdm
# pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132

# ---- Global Parameters ----
num_age_groups = 6 # Number of age groups

# ---- Epidemic Model ----
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


# ---- ML Models ----
class EpidemicMLP(L.LightningModule):
    """
    Predict the unobserved epidemic trajectory from the first
    n_observed_days of each simulation.

    Expected batch shapes:
        x: (batch_size, n_observed_days, num_features)
        y: (batch_size, n_forecast_days, num_features)
    """

    def __init__(
        self,
        n_observed_days: int,
        n_forecast_days: int,
        num_features: int,
        hidden_sizes: tuple[int, ...]=(256, 256),
        learning_rate: float=1e-3
    ):
        super().__init__()
        self.save_hyperparameters()
        input_size = (n_observed_days * num_features)
        output_size = (n_forecast_days * num_features)
        layer_sizes = (input_size, *hidden_sizes, output_size)
        layers = []
        for layer_idx in range(len(layer_sizes) - 2):
            layers.extend([
                nn.Linear(
                    layer_sizes[layer_idx],
                    layer_sizes[layer_idx + 1]
                ),
                nn.ReLU()
            ])
        layers.append(
            nn.Linear(
                layer_sizes[-2],
                layer_sizes[-1]
            )
        )
        self.network = nn.Sequential(*layers)
        self.loss_function = nn.MSELoss()

    def forward(self, x):
        batch_size = x.size(0)

        # Flatten observed time points and features
        x = x.reshape(batch_size, -1)
        predictions = self.network(x)

        # Restore time and feature dimensions
        predictions = predictions.reshape(
            batch_size,
            self.hparams.n_forecast_days,
            self.hparams.num_features
        )
        return predictions

    def _shared_step(self, batch, stage):
        x, y = batch
        predictions = self(x)
        loss = self.loss_function(predictions, y)
        self.log(
            f"{stage}_loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            batch_size=x.size(0)
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch=batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch=batch, stage="val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch=batch, stage="test")

    def predict_step(self, batch, batch_idx):
        x = (batch[0] if isinstance(batch, (tuple, list)) else batch)
        return self(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


class EpidemicTransformer(L.LightningModule):
    """
    Predict the unobserved epidemic trajectory from the first
    n_observed_days using a Transformer encoder.

    Expected batch shapes:
        x: (batch_size, n_observed_days, num_features)
        y: (batch_size, n_forecast_days, num_features)
    """

    def __init__(
        self,
        n_observed_days: int,
        n_forecast_days: int,
        num_features: int,
        d_model: int=128,
        nhead: int=4,
        num_layers: int=2,
        dim_feedforward: int=256,
        dropout: float=0.1,
        learning_rate: float=1e-3
    ):
        super().__init__()
        self.save_hyperparameters()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        # Project the epidemic features for each day into
        # the Transformer embedding dimension.
        self.input_projection = nn.Linear(num_features, d_model)

        # Learned representation of the position of each observed day.
        self.positional_embedding = nn.Parameter(torch.zeros(1, n_observed_days, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False
        )

        # Convert the encoded epidemic into the complete
        # future trajectory.
        self.output_layer = nn.Linear(
            n_observed_days * d_model,
            n_forecast_days * num_features
        )
        self.loss_function = nn.MSELoss()

    def forward(self, x):
        batch_size = x.size(0)

        # Project daily epidemic features into embedding space:
        # (batch_size, n_observed_days, num_features)
        # ->
        # (batch_size, n_observed_days, d_model)
        x = self.input_projection(x)

        # Add the position of each observed day
        x = (x + self.positional_embedding)

        # Apply self-attention across the observed days
        x = self.transformer_encoder(x)

        # Flatten all observed days and embeddings:
        # (batch_size, n_observed_days, d_model)
        # ->
        # (batch_size, n_observed_days * d_model)
        x = torch.flatten(x, start_dim=1)

        # Predict the complete future trajectory
        predictions = self.output_layer(x)

        # Restore forecast-day and feature dimensions
        predictions = predictions.reshape(
            batch_size,
            self.hparams.n_forecast_days,
            self.hparams.num_features
        )
        return predictions

    def _shared_step(self, batch, stage):
        x, y = batch
        predictions = self(x)
        loss = self.loss_function(predictions, y)
        self.log(
            f"{stage}_loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            batch_size=x.size(0)
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch=batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch=batch, stage="val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch=batch, stage="test")

    def predict_step(self, batch, batch_idx):
        x = (batch[0] if isinstance(batch, (tuple, list)) else batch)
        return self(x)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=1e-5
        )


if __name__ == "__main__":
    print("\n# ---- Create_SIRV_Training_Data.py Start ---- #")
    L.seed_everything(314159, workers=True)

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

    # ---- Run Epidemic Model ----
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

    # ---- Instantiate ML model ----
    trajectory = np.concatenate([epi["S"], epi["I"], epi["R"], epi["V"], epi["D"]], axis=2).astype(np.float32)
    n_observed_days = 14
    n_forecast_days = (trajectory.shape[1] - n_observed_days)
    population_scale = float(np.sum([1060, 1265, 1642, 4857, 3312, 2477]) * 10**3)
    trajectory = (trajectory / population_scale)
    x = trajectory[:, :n_observed_days, :]
    y = trajectory[:, n_observed_days:, :]
    print(x.shape) # (num_runs, n_observed_days, num_age_groups*5)
    print(y.shape) # (num_runs, t_max - n_observed_days, num_age_groups*5)
    model_architecture = "Transformer"
    if model_architecture == "MLP":
        model = EpidemicMLP(
            n_observed_days=n_observed_days,
            n_forecast_days=n_forecast_days,
            num_features=num_age_groups * 5,
            hidden_sizes=(256, 256),
            learning_rate=1e-3
        )
    elif model_architecture == "Transformer":
        model = EpidemicTransformer(
            n_observed_days=n_observed_days,
            n_forecast_days=n_forecast_days,
            num_features=num_age_groups * 5,
            d_model=64,
            nhead=4,
            num_layers=1,
            dim_feedforward=128,
            dropout=0.0,
            learning_rate=1e-4
        )
    else:
        raise ValueError(f"Unknown model architecture: {model_architecture}")

    print(f"Selected ML architecture: {model_architecture}")

    # ---- Convert Data to PyTorch Tensors ----
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    dataset = TensorDataset(x, y)

    # ---- Split Dataset ----
    num_samples = len(dataset)
    num_train = int(0.70 * num_samples)
    num_validation = int(0.15 * num_samples)
    num_test = (num_samples - num_train - num_validation)
    split_generator = torch.Generator().manual_seed(314159)
    train_dataset, validation_dataset, test_dataset = random_split(
        dataset=dataset,
        lengths=[num_train, num_validation, num_test],
        generator=split_generator
    )
    print(f"Training trajectories: {len(train_dataset)}")
    print(f"Validation trajectories: {len(validation_dataset)}")
    print(f"Test trajectories: {len(test_dataset)}")

    # ---- Create Data Loaders ----
    batch_size = 128
    cpu_workers = 0 # Data is already in memory so dont need os.cpu_count() - 1
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cpu_workers,
        persistent_workers=cpu_workers > 0,
        pin_memory=torch.cuda.is_available()
    )
    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cpu_workers,
        persistent_workers=cpu_workers > 0,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cpu_workers,
        persistent_workers=cpu_workers > 0,
        pin_memory=torch.cuda.is_available()
    )

    # ---- Train ML Model ----
    trainer = L.Trainer(
        max_epochs=100,
        accelerator="auto",
        devices="auto",
        deterministic=True,
        log_every_n_steps=10
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader
    )

    # ---- Test ML Model ----
    test_results = trainer.test(model=model, dataloaders=test_loader)
    print("Test results:")
    print(test_results)

    # ---- Run ML Inference ----
    prediction_batches = trainer.predict(model=model, dataloaders=test_loader)
    predictions = torch.cat(prediction_batches, dim=0)
    predictions = predictions.cpu().numpy()
    print(f"Predicted trajectory shape: {predictions.shape}")

    # ---- Collect Test Inputs and Targets ----
    observed_input_batches = []
    true_target_batches = []
    for input_batch, target_batch in test_loader:
        observed_input_batches.append(input_batch)
        true_target_batches.append(target_batch)

    observed_inputs = torch.cat(observed_input_batches, dim=0).numpy()
    true_targets = torch.cat(true_target_batches, dim=0).numpy()
    observed_counts = (observed_inputs * population_scale)
    predicted_counts = (predictions * population_scale)
    true_counts = (true_targets * population_scale)

    # ---- Calculate Forecast Errors ----
    forecast_errors = (predicted_counts - true_counts)
    mae = np.mean(np.abs(forecast_errors))
    rmse = np.sqrt(np.mean(forecast_errors**2))
    print(f"Overall MAE: {mae:,.0f} people")
    print(f"Overall RMSE: {rmse:,.0f} people")

    # ---- Calculate Compartment Errors ----
    feature_slices = {
        "S": slice(0, 6),
        "I": slice(6, 12),
        "R": slice(12, 18),
        "V": slice(18, 24),
        "D": slice(24, 30)
    }
    for compartment_name, feature_slice in feature_slices.items():
        compartment_errors = (predicted_counts[:, :, feature_slice] - true_counts[:, :, feature_slice])
        compartment_mae = np.mean(np.abs(compartment_errors))
        compartment_rmse = np.sqrt(np.mean(compartment_errors**2))
        print(f"{compartment_name} MAE: {compartment_mae:,.0f} people")
        print(f"{compartment_name} RMSE: {compartment_rmse:,.0f} people")

    # ---- Check Forecast Validity ----
    num_negative_predictions = np.count_nonzero(predicted_counts < 0)
    minimum_prediction = predicted_counts.min()
    print(f"Negative predictions: {num_negative_predictions:,}")
    print(f"Minimum predicted population: {minimum_prediction:,.1f}")

    # ---- Plot Median and Central 95% Forecast Intervals ----
    # Sum infectious populations across all six age groups
    observed_total_I = observed_counts[:, :, 6:12].sum(axis=2)
    true_total_I = true_counts[:, :, 6:12].sum(axis=2)
    predicted_total_I = predicted_counts[:, :, 6:12].sum(axis=2)

    # Calculate the median and central 95% intervals across
    # the held-out stochastic epidemic trajectories
    observed_lower, observed_median, observed_upper = np.percentile(
        observed_total_I, q=[2.5, 50.0, 97.5], axis=0
    )
    true_lower, true_median, true_upper = np.percentile(
        true_total_I, q=[2.5, 50.0, 97.5], axis=0
    )
    predicted_lower, predicted_median, predicted_upper = np.percentile(
        predicted_total_I, q=[2.5, 50.0, 97.5], axis=0
    )
    observed_time = np.arange(n_observed_days)
    forecast_time = np.arange(n_observed_days, n_observed_days + n_forecast_days)
    fig, ax = plt.subplots(figsize=(9, 4), facecolor="white")

    # Observed epidemic interval
    ax.fill_between(
        observed_time,
        observed_lower,
        observed_upper,
        color="black",
        alpha=0.15,
        linewidth=0,
        label="Observed central 95% interval"
    )

    ax.plot(
        observed_time,
        observed_median,
        color="black",
        linewidth=1.5,
        label="Observed median"
    )

    # True future epidemic interval
    ax.fill_between(
        forecast_time,
        true_lower,
        true_upper,
        color="black",
        alpha=0.15,
        linewidth=0,
        label="True central 95% interval"
    )

    ax.plot(
        forecast_time,
        true_median,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="True future median"
    )

    # Predicted future epidemic interval
    ax.fill_between(
        forecast_time,
        predicted_lower,
        predicted_upper,
        color="red",
        alpha=0.25,
        linewidth=0,
        label=f"{model_architecture} central 95% interval"
    )

    ax.plot(
        forecast_time,
        predicted_median,
        color="red",
        linewidth=1.5,
        label=f"{model_architecture} predicted median"
    )

    # Forecast begins after the final observed time point
    ax.axvline(
        n_observed_days - 1,
        color="grey",
        linestyle=":",
        linewidth=1.0,
        label="Forecast origin"
    )
    ax.set_xlabel("Time, $t$ (Days)")
    ax.set_ylabel("Total Infectious Population, $I(t)$")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    plt.show()
    
    print("# ---- Program End. ---- #\n")