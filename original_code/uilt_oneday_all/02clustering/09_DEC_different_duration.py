# -*- coding: utf-8 -*-
"""
@Time    : 2026/7/4 13:20
@Author  : HuangZhiyong
@email   : 1524338616@qq.com
@File    : 09_DEC_different_duration.py
@Function:
    Run per-event DEC clustering for different pre-event duration windows.

    The 24 h result already exists in the previous all-event DEC analysis, so
    this script skips 24 h by default. For each duration, it:
        1. Reads each event-duration recommended K from the output of
           08_DEC_different_duration_kneed.py.
        2. Selects the last N hours of STFT samples from every event.
        3. Loads the pretrained all-event encoder / autoencoder / decoder.
        4. Extracts latent features with the encoder.
        5. Initializes DEC cluster centers by KMeans, same as 01_DEC.py.
        6. Fine-tunes DEC with KLD + reconstruction loss, same as 01_DEC.py.
        7. Saves event-level npz files for later visualization.

    Output layout:
        ./result_different_duration_DEC_per_event/
            duration_01h/DEC_result/all_parement/event1_all_parement.npz
            duration_02h/DEC_result/all_parement/event1_all_parement.npz
            ...
"""

from __future__ import annotations

import gc
import glob
import os
import random
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from tensorflow.keras import backend as K
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import Input, InputSpec, Layer
from tensorflow.keras.models import Model

import Model_AE


# =========================================================
# 1. TensorFlow / path configuration
# =========================================================
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

USE_MIXED_PRECISION = False
if USE_MIXED_PRECISION:
    mixed_precision.set_global_policy("mixed_float16")
else:
    mixed_precision.set_global_policy("float32")


def set_seed(seed: int = 87) -> None:
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(87)

physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except Exception as exc:
        print("Error setting memory growth:", exc)


SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = Path(r"D:\Pycharm_Project\Find_precursor_signs\ILLgarben")
TRAIN_FILE_GLOB = str(PROJECT_ROOT / r"data\STFT_oneday_all\Z\*.npz")

AE_RESULT_ROOT = SCRIPT_DIR / "result_all_event"
GLOBAL_ENCODER_WEIGHTS = AE_RESULT_ROOT / r"model_save\all_events_Pretrain_encoder.h5"
GLOBAL_AUTOENCODER_WEIGHTS = AE_RESULT_ROOT / r"model_save\all_events_Pretrain_autoencoder.h5"
GLOBAL_DECODER_WEIGHTS = AE_RESULT_ROOT / r"model_save\all_events_Pretrain_decoder.h5"

K_LOOKUP_PATH = (
    SCRIPT_DIR
    / "result"
    / "different_duration_kneed"
    / "dec_k_lookup_by_event_duration.csv"
)

RESULT_ROOT = SCRIPT_DIR / "result_different_duration_DEC_per_event"


# =========================================================
# 2. Analysis configuration
# =========================================================
INPUT_SHAPE = (64, 512, 1)

# None means: read durations from K_LOOKUP_PATH and skip EXCLUDE_DURATIONS.
# You can also set a list manually, e.g. [1, 2, 4, 8, 12, 18].
DURATION_HOURS: list[int] | None = None
EXCLUDE_DURATIONS = {24}

# Read K from the event-duration lookup. If a pair is absent, DEFAULT_K is used.
K_COLUMN = "recommended_k"
DEFAULT_K = 6

CONFIG = {
    "gamma": 0.1,
    "batch_size": 256,
    "maxiter": 10000,
    "tol": 0.0001,
    "update_interval": 250,
    "save_interval": 250,
}

# If True, a completed duration is skipped. Delete that duration folder or set
# this to False if you want to rerun it.
SKIP_EXISTING = True

# Save spectral/reconstruction only when downstream plots need them. They are
# large, so the default is compact npz with x_file/feature/labels/probabilities.
SAVE_SPECTRAL = False
SAVE_RECONSTRUCTION = False

PREDICT_BATCH_SIZE = 256
MAX_TSNE_SAMPLES = 20000
RUN_TSNE = True


plt.rcParams.update(
    {
        "font.size": 12,
        "font.family": "Arial",
        "axes.unicode_minus": False,
    }
)
plt.ioff()


# =========================================================
# 3. Helpers: event/time parsing and STFT loading
# =========================================================
@dataclass
class DurationDataset:
    x_all: np.ndarray
    x_file_all: np.ndarray
    event_all: np.ndarray
    event_slices: dict[str, np.ndarray]


def natural_event_key(event_name: str) -> int:
    match = re.search(r"event(\d+)", str(event_name), flags=re.IGNORECASE)
    return int(match.group(1)) if match else 10**9


def extract_event_name_from_paths(files_array: np.ndarray, fallback_file: str) -> str:
    text = str(files_array[0]) if len(files_array) else str(fallback_file)
    match = re.search(r"event\d+", text, flags=re.IGNORECASE)
    if match:
        return match.group(0).lower()

    match = re.search(r"event\d+", str(fallback_file), flags=re.IGNORECASE)
    if match:
        return match.group(0).lower()

    return Path(fallback_file).stem.lower()


def parse_time_from_wave_file(wave_file: object) -> datetime | None:
    match = re.search(r"(\d{8}T\d{6})", str(wave_file))
    if match is None:
        return None
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")


def select_last_duration_indices(x_file: np.ndarray, duration_hours: int) -> np.ndarray:
    times = np.asarray([parse_time_from_wave_file(item) for item in x_file], dtype=object)
    if any(item is None for item in times):
        raise ValueError("Failed to parse timestamps from some waveform names.")

    end_time = max(times)
    start_time = end_time - timedelta(hours=float(duration_hours))
    keep = np.asarray([(item > start_time) and (item <= end_time) for item in times])

    return np.where(keep)[0]


def clean_and_normalize_stft(
    x: np.ndarray,
    eps: float = 1e-8,
    clip_percentile: float = 99.5,
) -> np.ndarray:
    """
    Clean and normalize STFT data consistently with the AE pretraining pipeline.
    """
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    finite_values = x[np.isfinite(x)]
    if finite_values.size > 0:
        high = np.percentile(finite_values, clip_percentile)
        low = np.percentile(finite_values, 100.0 - clip_percentile)
        if np.isfinite(high) and np.isfinite(low) and high > low:
            x = np.clip(x, low, high)

    if np.nanmin(x) >= 0:
        x = np.log1p(x)

    x_min = np.min(x, axis=(1, 2), keepdims=True)
    x_max = np.max(x, axis=(1, 2), keepdims=True)
    x = (x - x_min) / (x_max - x_min + eps)

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.astype(np.float32)


def list_event_stft_files() -> list[Path]:
    train_files = sorted(
        [Path(item) for item in glob.glob(TRAIN_FILE_GLOB)],
        key=lambda item: natural_event_key(item.name),
    )
    if not train_files:
        raise FileNotFoundError(f"No STFT files found: {TRAIN_FILE_GLOB}")
    return train_files


def load_event_duration_stft(stft_file: Path, duration_hours: int) -> DurationDataset:
    """
    Load the last duration_hours of one event.

    The returned structure keeps the same keys as the original duration dataset,
    so downstream visualization scripts can reuse the saved npz files.
    """
    data = np.load(stft_file, allow_pickle=True)
    if "stft" not in data or "files" not in data:
        raise KeyError(f"'stft' or 'files' not found in {stft_file}")

    x_train = data["stft"]
    x_file = data["files"]
    event_name = extract_event_name_from_paths(x_file, str(stft_file))

    idx = select_last_duration_indices(x_file, duration_hours)
    if len(idx) == 0:
        raise RuntimeError(f"{event_name}: no samples in last {duration_hours:g} h.")

    x_train = clean_and_normalize_stft(x_train[idx])
    x_file_all = x_file[idx]
    x_all = np.expand_dims(x_train, axis=-1).astype(np.float32)
    event_all = np.full(len(x_all), event_name, dtype=object)
    event_slices = {event_name: np.arange(len(x_all))}

    print(f"{event_name}: selected {len(x_all)} samples in last {duration_hours:g} h.")

    del data, x_train, x_file
    gc.collect()

    return DurationDataset(
        x_all=x_all,
        x_file_all=x_file_all,
        event_all=event_all,
        event_slices=event_slices,
    )


def load_duration_k_lookup() -> pd.DataFrame:
    if not K_LOOKUP_PATH.exists():
        raise FileNotFoundError(
            f"K lookup not found: {K_LOOKUP_PATH}\n"
            "Run 08_DEC_different_duration_kneed.py first."
        )

    lookup = pd.read_csv(K_LOOKUP_PATH)
    if "event" not in lookup.columns:
        raise KeyError(f"'event' not found in {K_LOOKUP_PATH}")
    if "duration_hours" not in lookup.columns:
        raise KeyError(f"'duration_hours' not found in {K_LOOKUP_PATH}")
    if K_COLUMN not in lookup.columns:
        raise KeyError(f"'{K_COLUMN}' not found in {K_LOOKUP_PATH}")

    lookup["event"] = lookup["event"].astype(str).str.lower()
    lookup["duration_hours"] = lookup["duration_hours"].astype(int)
    lookup[K_COLUMN] = lookup[K_COLUMN].astype(int)

    return lookup


def get_durations_to_run(lookup: pd.DataFrame) -> list[int]:
    if DURATION_HOURS is None:
        durations = sorted(lookup["duration_hours"].unique().astype(int).tolist())
    else:
        durations = [int(item) for item in DURATION_HOURS]

    durations = [item for item in durations if item not in EXCLUDE_DURATIONS]
    return durations


def get_k_for_event_duration(
    lookup: pd.DataFrame,
    event_name: str,
    duration_hours: int,
) -> int:
    event_name = str(event_name).lower()
    match = lookup.loc[
        (lookup["event"] == event_name)
        & (lookup["duration_hours"] == int(duration_hours))
    ]
    if match.empty:
        print(
            f"[WARN] {event_name}, duration={duration_hours:g} h absent from K lookup. "
            f"Use DEFAULT_K={DEFAULT_K}."
        )
        return int(DEFAULT_K)
    return int(match.iloc[0][K_COLUMN])


# =========================================================
# 4. Model and DEC layer
# =========================================================
class ClusteringLayer(Layer):
    """DEC clustering layer using Student's t-distribution."""

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if "input_shape" not in kwargs and "input_dim" in kwargs:
            kwargs["input_shape"] = (kwargs.pop("input_dim"),)

        super().__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))

        self.clusters = self.add_weight(
            shape=(self.n_clusters, input_dim),
            initializer="glorot_uniform",
            name="clusters",
        )

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        self.built = True

    def call(self, inputs, **kwargs):
        q = 1.0 / (
            1.0
            + (
                K.sum(
                    K.square(K.expand_dims(inputs, axis=1) - self.clusters),
                    axis=2,
                )
                / self.alpha
            )
        )
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {"n_clusters": self.n_clusters}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def target_distribution(q: np.ndarray) -> np.ndarray:
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T


def build_pretrained_dec_models(n_clusters: int):
    """
    Load pretrained encoder/autoencoder/decoder and build the full DEC model,
    following 01_DEC.py:
        clustering_layer = ClusteringLayer(...)(encoder.output)
        DECmodel = Model(autoencoder.input, [clustering_layer, autoencoder.output])
        loss = [kld, mse]
    """
    for weight_path in [
        GLOBAL_ENCODER_WEIGHTS,
        GLOBAL_AUTOENCODER_WEIGHTS,
        GLOBAL_DECODER_WEIGHTS,
    ]:
        if not weight_path.exists():
            raise FileNotFoundError(f"Missing pretrained AE weight: {weight_path}")

    encoder, decoder, autoencoder = Model_AE.CAE(
        input_shape=INPUT_SHAPE,
        filters=[8, 16, 32, 64, 128],
        encoding_filters=32,
        set_seed=[True, 87],
        summary=False,
    )

    encoder.load_weights(str(GLOBAL_ENCODER_WEIGHTS))
    autoencoder.load_weights(str(GLOBAL_AUTOENCODER_WEIGHTS))
    decoder.load_weights(str(GLOBAL_DECODER_WEIGHTS))

    clustering_layer = ClusteringLayer(n_clusters=n_clusters, name="clustering")(encoder.output)
    dec_model = Model(
        inputs=autoencoder.input,
        outputs=[clustering_layer, autoencoder.output],
        name=f"DEC_duration_K{n_clusters}",
    )
    q_model = Model(
        inputs=dec_model.input,
        outputs=dec_model.get_layer(name="clustering").output,
        name=f"DEC_duration_K{n_clusters}_q_only",
    )

    dec_model.compile(
        loss=["kld", "mse"],
        loss_weights=[CONFIG["gamma"], 1.0],
        optimizer="adam",
    )

    return encoder, decoder, autoencoder, dec_model, q_model


def predict_array_in_batches(model, x: np.ndarray, batch_size: int = PREDICT_BATCH_SIZE) -> np.ndarray:
    outputs = []
    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))
        pred = model(x[start:end], training=False).numpy()
        outputs.append(pred.astype(np.float32))
    return np.concatenate(outputs, axis=0)


def initialize_dec_with_kmeans(dec_model: Model, features: np.ndarray, n_clusters: int):
    """
    Initialize DEC cluster centers in the same way as 01_DEC.py:
        features -> KMeans(n_init=20) -> y_pred -> set clustering weights.
    """
    print("Initializing DEC centers by KMeans, same as 01_DEC.py...")
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(features)
    y_pred_last = np.copy(y_pred)
    dec_model.get_layer(name="clustering").set_weights([kmeans.cluster_centers_])

    print("Initial cluster counts:")
    for label, count in zip(*np.unique(y_pred, return_counts=True)):
        print(f"Cluster {int(label)}: {int(count)}")

    return y_pred, y_pred_last, kmeans.cluster_centers_.astype(np.float32)


def finetune_dec_like_01(
    dec_model: Model,
    q_model: Model,
    x_train: np.ndarray,
    y_pred_last: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, int]:
    """
    Follow the 01_DEC.py DEC refinement loop:
        q = DECmodel.predict(x_train)[0]
        p = target_distribution(q)
        y_pred = argmax(q)
        stop if delta_label < tol
        train_on_batch(x_train[idx], [p[idx], x_train[idx]])
    """
    print("Fine-tuning DEC with the same q -> p -> train_on_batch loop as 01_DEC.py...")

    loss_history = []
    index = 0
    index_array = np.arange(x_train.shape[0])
    p = None
    q = None
    y_pred = np.copy(y_pred_last)
    final_iter = 0

    for ite in range(int(CONFIG["maxiter"])):
        final_iter = ite

        if ite % int(CONFIG["update_interval"]) == 0:
            if len(x_train) > 15000:
                q1 = predict_array_in_batches(q_model, x_train[:15000])
                q2 = predict_array_in_batches(q_model, x_train[15000:])
                q = np.concatenate((q1, q2), axis=0)
            else:
                q = predict_array_in_batches(q_model, x_train)

            p = target_distribution(q)
            y_pred = q.argmax(axis=1)

            delta_label = (
                np.sum(y_pred != y_pred_last).astype(np.float32)
                / y_pred.shape[0]
            )
            y_pred_last = np.copy(y_pred)

            last_loss = loss_history[-1] if loss_history else None
            print(f"Iter {ite}: delta_label={delta_label:.6f}, loss={last_loss}")

            if ite > 0 and delta_label < float(CONFIG["tol"]):
                print(f"delta_label {delta_label:.6f} < tol {CONFIG['tol']}")
                break

        if p is None:
            raise RuntimeError("Target distribution p has not been initialized.")

        idx = index_array[
            index * int(CONFIG["batch_size"]): min(
                (index + 1) * int(CONFIG["batch_size"]),
                x_train.shape[0],
            )
        ]

        loss = dec_model.train_on_batch(x=x_train[idx], y=[p[idx], x_train[idx]])
        loss_history.append(loss)

        index = (
            index + 1
            if (index + 1) * int(CONFIG["batch_size"]) <= x_train.shape[0]
            else 0
        )

        if ite % 100 == 0:
            gc.collect()

    if q is None:
        q = predict_array_in_batches(q_model, x_train)
        y_pred = q.argmax(axis=1)

    loss_history = np.asarray(loss_history)
    if loss_history.ndim == 2 and loss_history.shape[1] == 3:
        df_loss = pd.DataFrame(
            loss_history,
            columns=["total_loss", "cluster_loss", "reconstruction_loss"],
        )
    else:
        df_loss = pd.DataFrame(loss_history)
    return q.astype(np.float32), y_pred.astype(int), df_loss, int(final_iter)


def compute_distances_to_centers(features: np.ndarray, cluster_centers: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(
        features[:, np.newaxis, :] - cluster_centers[np.newaxis, :, :],
        axis=2,
    )
    return distances.astype(np.float32)


# =========================================================
# 5. Evaluation, plotting, and saving
# =========================================================
def evaluate_clustering(features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    result = {
        "silhouette_score": np.nan,
        "calinski_harabasz_score": np.nan,
        "davies_bouldin_score": np.nan,
    }
    if len(np.unique(labels)) < 2:
        return result

    if len(features) > 20000:
        rng = np.random.default_rng(87)
        idx = rng.choice(len(features), size=20000, replace=False)
        eval_features = features[idx]
        eval_labels = labels[idx]
    else:
        eval_features = features
        eval_labels = labels

    try:
        result["silhouette_score"] = float(silhouette_score(eval_features, eval_labels))
    except Exception as exc:
        print(f"[WARN] silhouette_score failed: {exc}")

    try:
        result["calinski_harabasz_score"] = float(calinski_harabasz_score(eval_features, eval_labels))
    except Exception as exc:
        print(f"[WARN] calinski_harabasz_score failed: {exc}")

    try:
        result["davies_bouldin_score"] = float(davies_bouldin_score(eval_features, eval_labels))
    except Exception as exc:
        print(f"[WARN] davies_bouldin_score failed: {exc}")

    return result


def plot_loss_curve(
    df_loss: pd.DataFrame,
    save_path: Path,
    duration_hours: int,
    n_clusters: int,
    event_name: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for col in df_loss.columns:
        ax.plot(df_loss[col], label=col)
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Loss")
    title_prefix = f"{event_name}: " if event_name else ""
    ax.set_title(f"{title_prefix}DEC loss, last {duration_hours:g} h, K={n_clusters}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_pca_scatter(
    features: np.ndarray,
    labels: np.ndarray,
    save_path: Path,
    duration_hours: int,
    n_clusters: int,
    event_name: str | None = None,
):
    pca = PCA(n_components=2, random_state=87)
    redu_pca = pca.fit_transform(features).astype(np.float32)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        redu_pca[:, 0],
        redu_pca[:, 1],
        c=labels,
        cmap="tab10",
        s=8,
        alpha=0.65,
    )
    fig.colorbar(scatter, ax=ax, label="Cluster label")
    title_prefix = f"{event_name}: " if event_name else ""
    ax.set_title(f"{title_prefix}DEC, last {duration_hours:g} h, K={n_clusters}")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    return redu_pca


def plot_tsne_scatter(
    features: np.ndarray,
    labels: np.ndarray,
    save_path: Path,
    duration_hours: int,
    n_clusters: int,
    event_name: str | None = None,
):
    if len(features) > MAX_TSNE_SAMPLES:
        rng = np.random.default_rng(87)
        idx = rng.choice(len(features), size=MAX_TSNE_SAMPLES, replace=False)
        features_vis = features[idx]
        labels_vis = labels[idx]
    else:
        features_vis = features
        labels_vis = labels

    redu_tsne = TSNE(
        n_components=2,
        perplexity=40,
        random_state=74,
        init="pca",
        learning_rate="auto",
    ).fit_transform(features_vis)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        redu_tsne[:, 0],
        redu_tsne[:, 1],
        c=labels_vis,
        cmap="tab10",
        s=8,
        alpha=0.65,
    )
    fig.colorbar(scatter, ax=ax, label="Cluster label")
    title_prefix = f"{event_name}: " if event_name else ""
    ax.set_title(f"{title_prefix}t-SNE, last {duration_hours:g} h, K={n_clusters}")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def make_duration_dirs(duration_hours: int) -> dict[str, Path]:
    duration_root = RESULT_ROOT / f"duration_{duration_hours:02d}h"
    save_dir = duration_root / "DEC_result"
    dirs = {
        "duration_root": duration_root,
        "save_dir": save_dir,
        "image": save_dir / "image",
        "scatter": save_dir / "image" / "scatter",
        "loss": save_dir / "image" / "loss",
        "all_parameter": save_dir / "all_parement",
        "model_weight": save_dir / "model_weight",
        "loss_csv": save_dir / "loss_csv",
        "summary": save_dir / "summary",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def save_duration_outputs(
    dataset: DurationDataset,
    features: np.ndarray,
    redu_pca: np.ndarray,
    labels: np.ndarray,
    q: np.ndarray,
    distances: np.ndarray,
    cluster_centers: np.ndarray,
    reconstruction_all: np.ndarray,
    duration_hours: int,
    n_clusters: int,
    dirs: dict[str, Path],
) -> Path:
    empty_array = np.array([], dtype=np.float32)
    spectral_to_save = dataset.x_all if SAVE_SPECTRAL else empty_array
    reconstruction_to_save = reconstruction_all if SAVE_RECONSTRUCTION else empty_array

    global_output = {
        "x_file": dataset.x_file_all,
        "event": dataset.event_all,
        "duration_hours": duration_hours,
        "spectral": spectral_to_save,
        "feature": features,
        "redu": redu_pca,
        "redu_pca": redu_pca,
        "respectral": reconstruction_to_save,
        "y_pred_last": labels,
        "n_clusters": n_clusters,
        "probalities": q,
        "distances": distances,
        "cluster_centers": cluster_centers,
    }

    global_npz_path = (
        dirs["all_parameter"]
        / f"global_all_events_duration_{duration_hours:02d}h_all_parement_K{n_clusters}.npz"
    )
    np.savez(global_npz_path, **global_output)

    event_summary_rows = []
    for event_name, idx in dataset.event_slices.items():
        idx = np.asarray(idx)
        event_output = {
            "x_file": dataset.x_file_all[idx],
            "event": dataset.event_all[idx],
            "duration_hours": duration_hours,
            "spectral": spectral_to_save[idx] if SAVE_SPECTRAL else empty_array,
            "feature": features[idx],
            "redu": redu_pca[idx],
            "redu_pca": redu_pca[idx],
            "respectral": reconstruction_to_save[idx] if SAVE_RECONSTRUCTION else empty_array,
            "y_pred_last": labels[idx],
            "n_clusters": n_clusters,
            "probalities": q[idx],
            "distances": distances[idx],
            "cluster_centers": cluster_centers,
        }

        event_npz_path = dirs["all_parameter"] / f"{event_name}_all_parement.npz"
        np.savez(event_npz_path, **event_output)

        unique, counts = np.unique(labels[idx], return_counts=True)
        row = {
            "event": event_name,
            "duration_hours": duration_hours,
            "sample_count": int(len(idx)),
            "n_clusters_global": int(n_clusters),
            "npz_path": str(event_npz_path),
        }
        row.update({f"cluster_{int(k)}_count": int(v) for k, v in zip(unique, counts)})
        event_summary_rows.append(row)

    event_summary_df = pd.DataFrame(event_summary_rows)
    event_summary_df = event_summary_df.sort_values(
        by="event",
        key=lambda s: s.map(natural_event_key),
    )
    event_summary_df.to_excel(
        dirs["summary"] / f"event_cluster_count_summary_duration_{duration_hours:02d}h_K{n_clusters}.xlsx",
        index=False,
    )

    global_unique, global_counts = np.unique(labels, return_counts=True)
    global_cluster_summary = pd.DataFrame(
        {
            "cluster": global_unique.astype(int),
            "sample_count": global_counts.astype(int),
            "sample_ratio": global_counts / global_counts.sum(),
        }
    )
    global_cluster_summary.to_excel(
        dirs["summary"] / f"global_cluster_count_summary_duration_{duration_hours:02d}h_K{n_clusters}.xlsx",
        index=False,
    )

    return global_npz_path


def save_event_duration_outputs(
    dataset: DurationDataset,
    event_name: str,
    features: np.ndarray,
    redu_pca: np.ndarray,
    labels: np.ndarray,
    q: np.ndarray,
    distances: np.ndarray,
    cluster_centers: np.ndarray,
    reconstruction_all: np.ndarray,
    duration_hours: int,
    n_clusters: int,
    dirs: dict[str, Path],
) -> tuple[Path, dict[str, object]]:
    empty_array = np.array([], dtype=np.float32)
    spectral_to_save = dataset.x_all if SAVE_SPECTRAL else empty_array
    reconstruction_to_save = reconstruction_all if SAVE_RECONSTRUCTION else empty_array

    event_output = {
        "x_file": dataset.x_file_all,
        "event": dataset.event_all,
        "duration_hours": duration_hours,
        "spectral": spectral_to_save,
        "feature": features,
        "redu": redu_pca,
        "redu_pca": redu_pca,
        "respectral": reconstruction_to_save,
        "y_pred_last": labels,
        "n_clusters": n_clusters,
        "probalities": q,
        "distances": distances,
        "cluster_centers": cluster_centers,
    }

    event_npz_path = dirs["all_parameter"] / f"{event_name}_all_parement.npz"
    np.savez(event_npz_path, **event_output)

    unique, counts = np.unique(labels, return_counts=True)
    row = {
        "event": event_name,
        "duration_hours": int(duration_hours),
        "sample_count": int(len(labels)),
        "n_clusters": int(n_clusters),
        "npz_path": str(event_npz_path),
    }
    row.update({f"cluster_{int(k)}_count": int(v) for k, v in zip(unique, counts)})
    return event_npz_path, row


# =========================================================
# 6. DEC workflow for one event-duration pair
# =========================================================
def run_dec_for_event_duration(
    stft_file: Path,
    duration_hours: int,
    n_clusters: int,
) -> dict[str, object]:
    dirs = make_duration_dirs(duration_hours)
    event_name = extract_event_name_from_paths(np.array([], dtype=object), str(stft_file))
    event_npz_path = dirs["all_parameter"] / f"{event_name}_all_parement.npz"
    done_flag = dirs["summary"] / f"_DONE_{event_name}.txt"
    if SKIP_EXISTING and done_flag.exists() and event_npz_path.exists():
        print(f"\n[Skip] {event_name}, duration={duration_hours:g} h already finished: {done_flag}")
        return {
            "event": event_name,
            "duration_hours": duration_hours,
            "n_clusters": n_clusters,
            "status": "skipped_existing",
            "npz_path": str(event_npz_path),
            "output_dir": str(dirs["duration_root"]),
        }

    print("\n" + "=" * 80)
    print(f"Run per-event DEC: {event_name}, last {duration_hours:g} h, K={n_clusters}")
    print("=" * 80)

    dataset = load_event_duration_stft(stft_file, duration_hours)
    x_train = dataset.x_all
    n_samples = len(x_train)
    if n_samples < n_clusters:
        adjusted_k = max(1, n_samples)
        print(
            f"[WARN] {event_name}: n_samples={n_samples} < K={n_clusters}. "
            f"Use K={adjusted_k} instead."
        )
        n_clusters = adjusted_k

    encoder, decoder, autoencoder, dec_model, q_model = build_pretrained_dec_models(
        n_clusters=n_clusters,
    )

    print("Extracting initial latent features with pretrained encoder...")
    initial_features = predict_array_in_batches(encoder, x_train)

    _, y_pred_last, _ = initialize_dec_with_kmeans(
        dec_model=dec_model,
        features=initial_features,
        n_clusters=n_clusters,
    )

    q, labels, df_loss, final_iter = finetune_dec_like_01(
        dec_model=dec_model,
        q_model=q_model,
        x_train=x_train,
        y_pred_last=y_pred_last,
    )

    model_path = dirs["model_weight"] / f"{event_name}_duration_{duration_hours:02d}h_DEC_model_K{n_clusters}.h5"
    dec_model.save_weights(str(model_path))

    loss_csv_path = dirs["loss_csv"] / f"{event_name}_duration_{duration_hours:02d}h_training_loss_K{n_clusters}.csv"
    df_loss.to_csv(loss_csv_path, index=False)
    plot_loss_curve(
        df_loss,
        dirs["loss"] / f"{event_name}_duration_{duration_hours:02d}h_loss_curve_K{n_clusters}.png",
        duration_hours,
        n_clusters,
        event_name,
    )

    print("Running final prediction and extracting final latent features...")
    q = predict_array_in_batches(q_model, x_train)
    labels = q.argmax(axis=1)

    features = predict_array_in_batches(encoder, x_train)
    reconstruction_all = (
        predict_array_in_batches(decoder, features)
        if SAVE_RECONSTRUCTION
        else np.array([], dtype=np.float32)
    )

    cluster_centers = dec_model.get_layer(name="clustering").get_weights()[0].astype(np.float32)
    distances = compute_distances_to_centers(features, cluster_centers)

    redu_pca = plot_pca_scatter(
        features,
        labels,
        dirs["scatter"] / f"{event_name}_duration_{duration_hours:02d}h_pca_scatter_K{n_clusters}.png",
        duration_hours,
        n_clusters,
        event_name,
    )
    if RUN_TSNE:
        plot_tsne_scatter(
            features,
            labels,
            dirs["scatter"] / f"{event_name}_duration_{duration_hours:02d}h_tsne_scatter_K{n_clusters}.png",
            duration_hours,
            n_clusters,
            event_name,
        )

    metrics = evaluate_clustering(features, labels)
    metrics.update(
        {
            "event": event_name,
            "duration_hours": int(duration_hours),
            "n_samples": int(n_samples),
            "n_clusters": int(n_clusters),
            "final_iteration": int(final_iter),
            "model_weight_path": str(model_path),
        }
    )
    metrics_df = pd.DataFrame([metrics])
    metrics_path = dirs["summary"] / f"{event_name}_duration_{duration_hours:02d}h_DEC_K{n_clusters}_clustering_metrics.xlsx"
    metrics_df.to_excel(metrics_path, index=False)

    event_npz_path, event_summary_row = save_event_duration_outputs(
        dataset=dataset,
        event_name=event_name,
        features=features,
        redu_pca=redu_pca,
        labels=labels,
        q=q,
        distances=distances,
        cluster_centers=cluster_centers,
        reconstruction_all=reconstruction_all,
        duration_hours=duration_hours,
        n_clusters=n_clusters,
        dirs=dirs,
    )

    done_flag.write_text(
        f"event={event_name}\n"
        f"duration_hours={duration_hours}\n"
        f"n_clusters={n_clusters}\n"
        f"n_samples={n_samples}\n"
        f"finished_at={datetime.now().isoformat(timespec='seconds')}\n"
        f"event_npz={event_npz_path}\n",
        encoding="utf-8",
    )

    # Release memory between event-duration runs.
    del dataset, encoder, decoder, autoencoder, dec_model, q_model
    del x_train, initial_features, q, labels, features, redu_pca, distances, cluster_centers, reconstruction_all
    tf.keras.backend.clear_session()
    gc.collect()

    row = {
        "event": event_name,
        "duration_hours": duration_hours,
        "n_clusters": n_clusters,
        "status": "finished",
        "n_samples": int(n_samples),
        "npz_path": str(event_npz_path),
        "metrics_path": str(metrics_path),
        "output_dir": str(dirs["duration_root"]),
    }
    row.update(event_summary_row)
    return row


def main() -> None:
    print("Today's date:", str(date.today()))
    print("TensorFlow version:", tf.__version__)
    print("Mixed precision policy:", mixed_precision.global_policy())
    print("Physical devices:", tf.config.list_physical_devices())
    print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))

    lookup = load_duration_k_lookup()
    durations = get_durations_to_run(lookup)
    event_files = list_event_stft_files()

    if not durations:
        raise RuntimeError("No durations to run. Check DURATION_HOURS and EXCLUDE_DURATIONS.")

    print("\nDurations to run:", durations)
    print("Events to run:", len(event_files))
    print("24 h is excluded by default because the 24 h DEC result already exists.")
    print("Each event-duration pair is clustered independently; cluster labels are not comparable across events.")

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    run_rows = []
    for duration_hours in durations:
        duration_rows = []
        for stft_file in event_files:
            event_name = extract_event_name_from_paths(np.array([], dtype=object), str(stft_file))
            n_clusters = get_k_for_event_duration(lookup, event_name, duration_hours)
            row = run_dec_for_event_duration(stft_file, duration_hours, n_clusters)
            run_rows.append(row)
            duration_rows.append(row)

        duration_summary = pd.DataFrame(duration_rows)
        if not duration_summary.empty and "event" in duration_summary.columns:
            duration_summary = duration_summary.sort_values(
                by="event",
                key=lambda s: s.map(natural_event_key),
            )
        duration_dirs = make_duration_dirs(duration_hours)
        duration_summary.to_excel(
            duration_dirs["summary"] / f"duration_{duration_hours:02d}h_per_event_DEC_run_summary.xlsx",
            index=False,
        )

    run_summary = pd.DataFrame(run_rows)
    if not run_summary.empty:
        sort_columns = [col for col in ["duration_hours", "event"] if col in run_summary.columns]
        if sort_columns:
            run_summary = run_summary.sort_values(
                by=sort_columns,
                key=lambda s: s.map(natural_event_key) if s.name == "event" else s,
            )
    run_summary_path = RESULT_ROOT / "per_event_different_duration_DEC_run_summary.xlsx"
    run_summary.to_excel(run_summary_path, index=False)

    print("\nDone.")
    print("Run summary:", run_summary_path)
    print("Output root:", RESULT_ROOT)


if __name__ == "__main__":
    main()
