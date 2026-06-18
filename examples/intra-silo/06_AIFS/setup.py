import math
from logging import Logger, getLogger

import numpy as np
import torch
from anemoi.datasets import open_dataset
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.interface import AnemoiModelInterface
from anemoi.utils.config import DotDict
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


# ----- Imposta GPU se presente, altrimenti CPU ----- #
def set_device():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug(f"Device: {device}")

    return device


# ----- Costruzione del config ----- #
def load_config(training_yaml, model_yaml):

    # seleziona il config di training e del modello e li unisce in uno solo
    training_cfg = OmegaConf.load(training_yaml)
    model_cfg = OmegaConf.load(model_yaml)
    config_omegaconf = OmegaConf.merge(training_cfg, model_cfg)
    OmegaConf.resolve(config_omegaconf)

    # trasforma l'omegaconf in dot dict cioè un oggetto che permette di accedere ai suoi attributi valori con la notazione punto
    config = DotDict(OmegaConf.to_container(config_omegaconf, resolve=True))

    # config è in formato DotDict che serve per AnemoiModelInterface
    # config_omegaconf è in formato OmegaConf che serve per IndexCollection
    return config, config_omegaconf


# ----- Costruzione del grafo ----- #
def load_graph(config):

    logger.debug("Caricamento del grafo...")
    graph_data = torch.load(config.files.graph, weights_only=False)
    logger.debug(f"   Nodi 'data': {graph_data['data'].num_nodes}")
    logger.debug(f"   Nodi 'hidden': {graph_data['hidden'].num_nodes}")

    return graph_data


def combined_statistics(config):
    # il numero di punti con cui è calcolata la media è #timestep del train-set * #nodi, o no? claudio aveva usato solo il # punti
    ds_tot = config.files.dataset["grids"]

    ds1 = open_dataset(ds_tot[0])
    ds2 = open_dataset(ds_tot[1])
    n1 = len(ds1) * ds1.shape[-1]
    n2 = len(ds2) * ds2.shape[-1]

    media1 = ds1.statistics["mean"]
    media2 = ds2.statistics["mean"]

    stdev1 = ds1.statistics["stdev"]
    stdev2 = ds2.statistics["stdev"]

    max1 = ds1.statistics["maximum"]
    max2 = ds2.statistics["maximum"]

    min1 = ds1.statistics["minimum"]
    min2 = ds2.statistics["minimum"]

    n_variabili_ds1 = len(ds1.name_to_index)
    n_variabili_ds2 = len(ds2.name_to_index)

    if n_variabili_ds1 != n_variabili_ds2:
        logger.debug("Errore: i due dataset non hanno lo stesso numero di variabili")

    n_var = n_variabili_ds1

    media = []
    stdev = []
    ex2 = []
    massimo = []
    minimo = []
    for i in range(n_var):

        a = (n1 * media1[i] + n2 * media2[i]) / (n1 + n2)
        media.append(a)

        b = (
            n1 * (stdev1[i] ** 2 + media1[i] ** 2)
            + n2 * (stdev2[i] ** 2 + media2[i] ** 2)
        ) / (n1 + n2)
        ex2.append(b)

        c = math.sqrt(max(0.0, ex2[i] - media[i] ** 2))
        stdev.append(c)

        d = max(max1[i], max2[i])
        massimo.append(d)

        e = min(min1[i], min2[i])
        minimo.append(e)

    return {
        "mean": np.array(media),
        "stdev": np.array(stdev),
        "minimum": np.array(minimo),
        "maximum": np.array(massimo),
    }


def load_data(config):

    logger.debug("Caricamento del dataset...")
    # il dataset vero e proprio
    ds = open_dataset(config.files.dataset)
    # dizionario di tutte le variabili: key=variable name, value=index {"10u": 0, "10v": 1, "2d": 2, "2t": 3, ...}
    name_to_index = ds.name_to_index
    # Se il dataset è una combinazione grids: (es. centralizzato EU+USA), ricalcoliamo le statistiche globali
    if "grids" in config.files.dataset:
        statistics = combined_statistics(config)
    else:
        statistics = ds.statistics

    logger.debug(
        f"   Timesteps: {len(ds)}, Variabili: {len(name_to_index)}, Gridpoints: {ds.shape[-1]}"
    )

    return ds, name_to_index, statistics


# ----- Costruzione degli indici ----- #
# Costruisce l'IndexCollection che mappa variabili a indici input/output.
# IndexCollection richiede OmegaConf (non DotDict) come primo argomento.
def build_indices(config_omegaconf, name_to_index):

    logger.debug("Costruzione indici dati...")
    data_indices = IndexCollection(config_omegaconf, name_to_index)

    num_input = len(data_indices.internal_model.input)
    num_output = len(data_indices.internal_model.output)
    logger.debug(f"   Variabili input modello: {num_input} (prognostic + forcing)")
    logger.debug(f"   Variabili output modello: {num_output} (prognostic + diagnostic)")

    # data_indices è un oggetto i cui attributi sono liste di indici delle variabili
    # è un oggetto più ricco di name_to_index, che è invece "rigido"
    return data_indices


# ----- Costruzione del modello ----- #
def build_model(
    config, graph_data, statistics, data_indices, device, checkpoint_path=None
):

    logger.debug("Costruzione modello con AnemoiModelInterface...")
    model = AnemoiModelInterface(
        config=config,
        graph_data=graph_data,
        statistics=statistics,
        data_indices=data_indices,
        metadata={},
        supporting_arrays={},
        truncation_data={},
    )

    # se il checkpoint_path non è nullo,carica i pesi pre-addestrati (nel finetuning dovrai specificare)
    if checkpoint_path is not None:
        logger.debug(f"   Caricamento checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        logger.debug("   Checkpoint caricato!")

    # in ogni caso mettiamo il modello su GPU se presente
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.debug(f"   Parametri addestrabili: {total_params:,}")

    return model


# ----- Dataset class ----- #
class AnemoiNativeDataset(Dataset):

    def __init__(self, ds, multistep=2, rollout=1, start_idx=0, end_idx=None):
        self.ds = ds  # ds = [t_0, t_1, t_2, ..., t_N], Ogni t ha shape: (43 vars, 1 ensemble, 40320 gridpoints)
        self.multistep = (
            multistep  # numero di timestep in input: 2, ovvero t_(-6) e t_0
        )
        self.rollout = rollout  # numero di step di predizione usando lo stesso campione (1 in pretraining, 12 in finetuning)
        self.start_idx = (
            start_idx  # indice di partenza del segmento di dataset considerato
        )
        self.end_idx = (
            end_idx if end_idx is not None else len(ds)
        )  # indice di fine del segmento di dataset considerato
        # numero di istanti temporali considerati nel segmento
        segment_len = self.end_idx - self.start_idx
        # numero di campioni validi in questo segmento
        self.num_samples = segment_len - multistep - rollout + 1

    def __len__(self):
        return self.num_samples

    # prende il pezzo di dataset che ti serve, cioè ritaglia un campione
    def __getitem__(self, idx):
        actual_idx = self.start_idx + idx
        steps = self.multistep + self.rollout
        chunk = self.ds[actual_idx : actual_idx + steps]
        chunk = np.array(chunk, dtype=np.float32)
        # (steps, vars, ensemble, gridpoints) → (steps, ensemble, gridpoints, vars)
        chunk = np.transpose(chunk, (0, 2, 3, 1))
        return torch.from_numpy(chunk)


# ----- costruzione dizionario {pezzo di dataset, indici inizio-fine di quel pezzo} ----- #
def split_dataset(ds, val_years=2, test_years=1, timesteps_per_year=1461):

    n_total = len(ds)  # 2005-2024 = 20 anni
    n_test = test_years * timesteps_per_year
    n_val = val_years * timesteps_per_year
    n_train = n_total - n_val - n_test

    splits = {
        "train": (0, n_train),  # 2005-2021 = 17 anni
        "val": (n_train + n_test, n_total),  # 2023-2024 = 2 anni
        "test": (n_train, n_train + n_test),  # 2022 = 1 anno
    }

    logger.debug("   Dataset split:")
    for name, (start, end) in splits.items():
        logger.debug(
            f"      {name:>5s}: indici [{start}, {end}) → {end - start} timestep (~{(end - start) / timesteps_per_year:.1f} anni)"
        )

    return splits


def split_dataset_finetuning(ds, val_years=2, test_years=1, timesteps_per_year=1461):

    n_total = len(ds)  # 2005-2024 = 20 anni
    n_test = test_years * timesteps_per_year
    n_val = val_years * timesteps_per_year
    n_train = n_total - n_val - n_test
    start_year_finetuning = timesteps_per_year * 11  # 2016

    splits = {
        "train": (start_year_finetuning, n_train),  # 2016-2021 = 6 anni
        "val": (n_train + n_test, n_total),  # 2023-2024 = 2 anni
        "test": (n_train, n_train + n_test),  # 2022 = 1 anno
    }

    logger.debug("   Dataset split:")
    for name, (start, end) in splits.items():
        logger.debug(
            f"      {name:>5s}: indici [{start}, {end}) → {end - start} timestep (~{(end - start) / timesteps_per_year:.1f} anni)"
        )

    return splits


# ----- Gestione del dataset a livello di batches -----#
def build_dataloader(
    ds,
    multistep,
    rollout,
    batch_size,
    device_type="cuda",
    start_idx=0,
    end_idx=None,
    shuffle=True,
):

    dataset = AnemoiNativeDataset(
        ds, multistep=multistep, rollout=rollout, start_idx=start_idx, end_idx=end_idx
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        # numero di sottoprocessi paralleli che PyTorch utilizza per estrarre i campioni dal tuo dataset e preparare i batch.
        num_workers=8,
        pin_memory=(device_type == "cuda"),
    )
    logger.debug(f"   Campioni: {len(dataset)}, Batch: {len(loader)}")
    return loader


# ----- Assegnazione pesi ----- #
# costruisce i pesi per la Weighted MSE Loss (riproduce forecaster.py → get_variable_scaling)
# e per le node weights (riproduce nodeweights.py → GraphNodeAttribute.weights)
def build_loss_weights(config, data_indices, graph_data, device):

    vls = config.training.variable_loss_scaling
    pls = config.training.pressure_level_scaler
    output_name_to_idx = data_indices.internal_model.output.name_to_index
    num_output = len(output_name_to_idx)

    # variable weights
    var_weights = torch.ones(num_output, dtype=torch.float32) * vls.default

    for name, idx in output_name_to_idx.items():
        split = name.split("_")
        if len(split) > 1 and split[-1].isdigit():
            # Variabile su livello di pressione (es. t_850)
            if split[0] in vls.get("pl", {}):
                level = int(split[-1])
                pl_scale = max(pls.minimum, pls.slope * level)
                var_weights[idx] = vls.pl[split[0]] * pl_scale
        else:
            # Variabile di superficie (es. sp, 10u)
            if name in vls.get("sfc", {}):
                var_weights[idx] = vls.sfc[name]

    var_weights = var_weights.to(device)

    logger.debug("   Variable loss weights:")
    for name, idx in sorted(output_name_to_idx.items(), key=lambda x: x[1]):
        logger.debug(f"      {name:>8s}: {var_weights[idx].item():.4f}")

    # node area weights
    node_attr = config.training.node_loss_weights
    node_weights = graph_data[config.graph.data][node_attr].squeeze().to(device)
    logger.debug(
        f"   Node area weights: shape={node_weights.shape}, "
        f"min={node_weights.min():.4f}, max={node_weights.max():.4f}"
    )

    return var_weights, node_weights


# ----- Loss function ----- #
# riproduce WeightedMSELoss.forward() + scale_by_node_weights() da anemoi/training/losses/mse.py e weightedloss.py.
def weighted_mse_loss(y_pred, y, var_weights, node_weights):

    err = torch.square(y_pred.float() - y.float())  # (y_pred - y)^2
    err = (
        err * var_weights
    )  # moltiplico ogni errore per il peso della variabile corrispondente
    err = torch.mean(
        err, dim=-1
    )  # media tra tutti gli elementi del tensore per ogni punto della griglia
    err = (
        err * node_weights
    )  # moltiplico ogni errore per il peso del nodo corrispondente
    return torch.sum(err) / torch.sum(node_weights.expand_as(err))


# ----- Gestione rollout ----- #
# Aggiorna l'input per il prossimo step autogressivo, (riproduce forecaster.py → advance_input)
def advance_input(x, y_pred, batch, rollout_step, multistep, data_indices):
    """
    1. Shift: [t-1, t] → [t, vuoto]
    2. Inserisce prognostic dalla predizione del modello
    3. Aggiorna forcing dal batch (timestep corretto)

    Args:
        x: input corrente, shape (bs, multistep, ensemble, grid, n_input)
        y_pred: predizione, shape (bs, ensemble, grid, n_output)
        rollout_step: indice del passo di rollout (0-based)
        data_indices: IndexCollection
    """
    x = x.roll(-1, dims=1)

    x[:, -1, :, :, data_indices.internal_model.input.prognostic] = y_pred[
        ..., data_indices.internal_model.output.prognostic
    ]

    x[:, -1, :, :, data_indices.internal_model.input.forcing] = batch[
        :, multistep + rollout_step, :, :, data_indices.internal_data.input.forcing
    ]

    return x


# ----- Validation ----- #
# esegue un loop di validazione senza aggiornare i pesi (riproduce forecaster.py → run_validation)
def run_validation(
    model,
    val_loader,
    var_weights,
    node_weights,
    input_idx,
    output_idx,
    multistep,
    device,
    rollout=1,
    data_indices=None,
):

    model.eval()
    val_loss = 0.0
    num_batches = 0
    use_amp = device.type == "cuda"

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            batch = model.pre_processors(batch, in_place=False)

            x = batch[:, :multistep, ..., input_idx]

            # Accumula loss su tutti i rollout step (come nel training)
            batch_loss = torch.zeros(1, dtype=batch.dtype, device=device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                for rstep in range(rollout):
                    y_pred = model(x)
                    y = batch[:, multistep + rstep, ..., output_idx]
                    step_loss = weighted_mse_loss(y_pred, y, var_weights, node_weights)
                    batch_loss = batch_loss + step_loss

                    # Avanza l'input per il prossimo step
                    if rstep < rollout - 1 and data_indices is not None:
                        x = advance_input(
                            x, y_pred, batch, rstep, multistep, data_indices
                        )

            val_loss += (batch_loss / rollout).item()
            num_batches += 1

    avg_val_loss = val_loss / max(num_batches, 1)
    return avg_val_loss


# ----- Validation LAM ----- #
# Identica a run_validation ma applica la boundary condition:
# dopo advance_input, i nodi del boundary domain vengono sovrascrittti
# con i valori ERA5 reali del batch (già normalizzati).
# Parametri aggiuntivi rispetto a run_validation:
#   boundary_mask : BoolTensor (N_grid,) — True dove il nodo è boundary
#   input_idx     : indici per selezionare le variabili input dal batch
def run_validation_lam(
    model,
    val_loader,
    var_weights,
    node_weights,
    input_idx,
    output_idx,
    multistep,
    device,
    boundary_mask,
    rollout=1,
    data_indices=None,
):

    model.eval()
    val_loss = 0.0
    num_batches = 0
    use_amp = device.type == "cuda"

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            batch = model.pre_processors(batch, in_place=False)

            x = batch[:, :multistep, ..., input_idx]

            batch_loss = torch.zeros(1, dtype=batch.dtype, device=device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                for rstep in range(rollout):
                    y_pred = model(x)
                    y = batch[:, multistep + rstep, ..., output_idx]
                    step_loss = weighted_mse_loss(y_pred, y, var_weights, node_weights)
                    batch_loss = batch_loss + step_loss

                    if rstep < rollout - 1 and data_indices is not None:
                        x = advance_input(
                            x, y_pred, batch, rstep, multistep, data_indices
                        )
                        # LAM boundary condition: boundary nodes ← ERA5 reale al passo t+1
                        next_ts = multistep + rstep + 1
                        x[:, -1, :, boundary_mask, :] = batch[
                            :, next_ts, :, boundary_mask, :
                        ][:, :, :, input_idx]

            val_loss += (batch_loss / rollout).item()
            num_batches += 1

    avg_val_loss = val_loss / max(num_batches, 1)
    return avg_val_loss


# ----- Climatologia (per ACC) — metodo WeatherBench2 ----- #
def compute_climatology(
    ds,
    start_idx,
    end_idx,
    output_idx,
    window_size=61,
    chunk_size=100,
    save_path="climatology.pt",
    spatial_mask=None,
):
    """Pre-calcola la climatologia smoothed seguendo la metodologia WeatherBench2.

    Usa una rolling window di `window_size` giorni (default 61 = ±30 giorni)
    con pesi triangolari linearmente decrescenti dal centro, come descritto
    in Jung & Leutbecher (2008) e WeatherBench2 (Rasp et al., 2024).

    Metodo 'fast' (equivalente a 'explicit' per la media):
      1. Calcola la media grezza per ogni (giorno_anno, ora) su tutti gli anni
      2. Applica smoothing triangolare lungo l'asse giorno_anno

    I valori sono in unità fisiche (raw zarr, non normalizzati).

    Args:
        ds: dataset anemoi
        start_idx: indice di inizio dei dati di training
        end_idx: indice di fine dei dati di training
        output_idx: indici delle variabili output nello spazio dati
        window_size: dimensione della finestra in giorni (deve essere dispari)
        chunk_size: timestep caricati per volta (per efficienza)

    Returns:
        climatology: dict[(doy, hour)] → np.array (ensemble, gridpoints, n_output)
    """
    import pandas as pd

    assert (
        window_size % 2 == 1
    ), f"window_size deve essere dispari, ricevuto {window_size}"
    half_window = window_size // 2  # 30

    logger.debug(f"   Calcolo climatologia WB2 (window={window_size} giorni)...")

    # ------------------------------------------------------------------
    # Step 1: Medie grezze per (doy, hour) su tutti gli anni di training
    # ------------------------------------------------------------------
    dates = ds.dates[start_idx:end_idx]
    dt_index = pd.DatetimeIndex(dates)
    doys = dt_index.dayofyear.values  # 1-366
    hours = dt_index.hour.values  # 0, 6, 12, 18

    clim_sum = {}
    clim_count = {}
    n_steps = end_idx - start_idx

    for cs in range(0, n_steps, chunk_size):
        ce = min(cs + chunk_size, n_steps)
        # Zarr shape: (chunk, n_vars, ensemble, gridpoints)
        data = np.array(ds[start_idx + cs : start_idx + ce], dtype=np.float32)
        # Seleziona output vars e trasponi: → (chunk, ensemble, gridpoints, n_output)
        data = data[:, output_idx, :, :]
        data = np.transpose(
            data, (0, 2, 3, 1)
        )  # (chunk, ensemble, gridpoints, n_output)
        if spatial_mask is not None:
            data = data[:, :, spatial_mask, :]  # (chunk, ensemble, N_masked, n_output)

        for j in range(ce - cs):
            key = (int(doys[cs + j]), int(hours[cs + j]))
            if key not in clim_sum:
                clim_sum[key] = np.zeros(data.shape[1:], dtype=np.float64)
                clim_count[key] = 0
            clim_sum[key] += data[j].astype(np.float64)
            clim_count[key] += 1

        if (cs + chunk_size) % 5000 < chunk_size:
            logger.debug(f"      {min(cs + chunk_size, n_steps)}/{n_steps} timestep...")

    # Medie grezze
    raw_clim = {}
    for key in clim_sum:
        raw_clim[key] = clim_sum[key] / clim_count[key]

    logger.debug(f"      Medie grezze: {len(raw_clim)} combinazioni (doy, ora)")

    # ------------------------------------------------------------------
    # Step 2: Smoothing triangolare (identico a WB2 create_window_weights)
    # ------------------------------------------------------------------
    # Kernel triangolare: [0, 1/30, 2/30, ..., 1, ..., 2/30, 1/30, 0]
    tri_weights = np.concatenate(
        [
            np.linspace(0, 1, half_window + 1),
            np.linspace(1, 0, half_window + 1)[1:],
        ]
    )
    # Non serve normalizzare a priori: dividiamo per la somma dei pesi usati

    unique_hours = sorted(set(h for _, h in raw_clim.keys()))

    climatology = {}
    for hour in unique_hours:
        for doy in range(1, 367):  # 1-366
            weighted_sum = None
            weight_total = 0.0

            for i in range(-half_window, half_window + 1):
                # Wrap-around periodico: giorno 366 → giorno 1
                neighbor_doy = ((doy - 1 + i) % 366) + 1
                w = tri_weights[i + half_window]

                if (neighbor_doy, hour) in raw_clim:
                    if weighted_sum is None:
                        weighted_sum = w * raw_clim[(neighbor_doy, hour)]
                    else:
                        weighted_sum += w * raw_clim[(neighbor_doy, hour)]
                    weight_total += w

            if weighted_sum is not None and weight_total > 0:
                climatology[(doy, hour)] = (weighted_sum / weight_total).astype(
                    np.float32
                )

    logger.debug(
        f"   Climatologia smoothed: {len(climatology)} combinazioni (doy, ora)"
    )

    torch.save(climatology, save_path)
    logger.debug(f"   Climatologia salvata in '{save_path}'")

    return climatology


def save_global_step(x):
    with open("last_global_step.txt", "w") as f:
        logger.debug(x, file=f)


def load_global_step():
    with open("last_global_step.txt", "r") as f:
        last_global_step = f.read()
    return int(last_global_step)


def compute_climatology_basic(
    ds,
    start_idx,
    end_idx,
    output_idx,
    chunk_size=100,
    save_path="climatology.pt",
    spatial_mask=None,
):

    import pandas as pd

    # Preparazione oggetti per calcolo climatologia
    dates = ds.dates[start_idx:end_idx]  # seleziono le date (che saranno del train set)
    dt_index = pd.DatetimeIndex(
        dates
    )  # scompone le date in formato 2015-07-15T12:00:00 in oggetti separati
    doys = dt_index.dayofyear.values  # 1-366
    hours = dt_index.hour.values  # 0, 6, 12, 18

    # conterrà la somma di tutti i valori di una certa variabile per uno specifico giorno/ora
    # key: (doy, hour) , sono 1461 timestep in un anno
    # value: np.array di shape (ensemble, gridpoints, n_output) cioè per ogni timestep c'è una griglia in cui i 20 anni sono sommati
    clim_sum = {}
    # stessa key, ma il value è quanti anni di dataset hai sommato per essa, cioè 17
    clim_count = {}
    # inizio e fine del train set, 24837 timestep
    n_steps = end_idx - start_idx

    for cs in range(0, n_steps, chunk_size):  # avanti a salti di 100 timestep

        ce = min(cs + chunk_size, n_steps)
        # Zarr shape: (chunk, n_vars, ensemble, gridpoints)
        # considera da         qui        a      qui
        data = np.array(ds[start_idx + cs : start_idx + ce], dtype=np.float32)

        # Seleziona output vars, le 30 variabili che predice il modello
        data = data[:, output_idx, :, :]

        # Trasponi: da (chunk, n_output,ensemble, gridpoints) a (chunk, ensemble, gridpoints, n_output), perchè però?
        data = np.transpose(data, (0, 2, 3, 1))

        # Se vuoi valutare solo su una certa zona applichi la mask, non serve per il modello già locale, ma per un globale
        if spatial_mask is not None:
            data = data[:, :, spatial_mask, :]  # (chunk, ensemble, N_masked, n_output)

        # Sul pezzo di dataset che hai ritagliato e maneggiato, calcola la climatologia.
        for j in range(ce - cs):
            # creo la key, che è tipo (doy, hour) ad esempio (196, 12)
            key = (int(doys[cs + j]), int(hours[cs + j]))
            # se la key è nuova inizializza a 0, solo per il primo giro
            if key not in clim_sum:
                clim_sum[key] = np.zeros(data.shape[1:], dtype=np.float64)
                clim_count[key] = 0
            # inserisci la griglia per quel giorno e ora
            clim_sum[key] += data[j].astype(np.float64)

            # aggiungi 1 all'anno, che in totale saranno 17
            clim_count[key] += 1

    # Riempiamo il dizionario, c = media dei valori nelgli anni
    climatology = {}

    for key in clim_sum:
        climatology[key] = (clim_sum[key] / clim_count[key]).astype(np.float32)

    torch.save(climatology, save_path)
    logger.debug(f"   Climatologia salvata in '{save_path}'")

    return climatology


# ----- Costruzione del dataset ----- #
def _compute_combined_statistics_deprecated(grids_config):
    """Calcola le statistiche combined su più dataset (Proposta 1: media pesata).

    Usa la formula della varianza pooled per stdev, media pesata per mean,
    e min/max elementwise. Serve quando config.files.dataset è un dict grids:
    (es. modello centralizzato EU+USA).

    Parameters
    ----------
    grids_config : list of dict
        Lista di config dict (ognuno con 'dataset': path) come specificato
        in files.dataset.grids nel YAML di training.

    Returns
    -------
    dict
        Statistiche combined: mean, stdev, minimum, maximum per variabile.
    """
    import numpy as np

    # Apre ogni sotto-dataset per leggerne le statistiche
    sub_datasets = [open_dataset(cfg) for cfg in grids_config]

    # Numero di gridpoints per dataset → usato come peso della media
    n_points = [ds_i.shape[-1] for ds_i in sub_datasets]
    n_total = sum(n_points)
    stats_list = [ds_i.statistics for ds_i in sub_datasets]

    # --- Media pesata per numero di gridpoints ---
    mean_combined = (
        sum(n_i * s["mean"] for n_i, s in zip(n_points, stats_list)) / n_total
    )

    # --- Stdev: formula della varianza pooled ---
    # E[X^2]_i = var_i + mean_i^2 = stdev_i^2 + mean_i^2
    # var_combined = weighted_mean(E[X^2]) - mean_combined^2
    # stdev_combined = sqrt(var_combined)
    e_x2 = (
        sum(
            n_i * (s["stdev"] ** 2 + s["mean"] ** 2)
            for n_i, s in zip(n_points, stats_list)
        )
        / n_total
    )
    stdev_combined = np.sqrt(np.maximum(e_x2 - mean_combined**2, 0.0))

    # --- Min/Max elementwise ---
    minimum_combined = np.minimum.reduce([s["minimum"] for s in stats_list])
    maximum_combined = np.maximum.reduce([s["maximum"] for s in stats_list])

    logger.debug(
        f"   Statistiche combined: {len(sub_datasets)} dataset, "
        f"nodi: {' + '.join(str(n) for n in n_points)} = {n_total}"
    )

    return {
        "mean": mean_combined,
        "stdev": stdev_combined,
        "minimum": minimum_combined,
        "maximum": maximum_combined,
    }
