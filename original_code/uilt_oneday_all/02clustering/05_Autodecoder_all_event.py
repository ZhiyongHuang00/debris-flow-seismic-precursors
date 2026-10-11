# -*- coding: utf-8 -*-
"""
Memory-safe all-event Autoencoder pretraining.

This version avoids concatenating one full-day STFT array from all events into
one giant in-memory tensor. Instead, it:

1. Scans only sample metadata first.
2. Splits train/validation on metadata level.
3. Streams STFT batches from disk with a Keras Sequence.
4. Encodes latent features event by event after training.

The output paths and model interface remain consistent with the previous
workflow so downstream code can keep working.
"""

from __future__ import annotations

import gc
import glob
import os
import random
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.utils import Sequence

import Model_AE


# =========================================================
# 1. TensorFlow / GPU configuration
# =========================================================
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

USE_MIXED_PRECISION = False

if USE_MIXED_PRECISION:
    mixed_precision.set_global_policy("mixed_float16")
else:
    mixed_precision.set_global_policy("float32")

print("Mixed precision policy:", mixed_precision.global_policy())
print("TensorFlow version:", tf.__version__)
print("Physical devices:", tf.config.list_physical_devices())
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))

physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except Exception as exc:
        print("Error setting memory growth:", exc)


def set_seed(seed: int = 87):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(87)


# =========================================================
# 2. Path configuration
# =========================================================
PROJECT_ROOT = Path(r"D:\Pycharm_Project\Find_precursor_signs\ILLgarben")
TRAIN_FILE_GLOB = str(PROJECT_ROOT / r"data\STFT_oneday_all\Z\*Z.npz")

RESULT_ROOT = Path(r".\result_all_event")
AE_LOG_DIR = RESULT_ROOT / "AE"
MODEL_SAVE_DIR = RESULT_ROOT / "model_save"
ALL_PARAMETER_DIR = RESULT_ROOT / "all_parameters"
EVENT_PARAMETER_DIR = RESULT_ROOT / "all_parameters_by_event"
FIGURE_LOSS_DIR = RESULT_ROOT / "Figure" / "loss"
RECON_FIGURE_DIR = RESULT_ROOT / "Figure" / "reconstruction_examples"

for d in [
    RESULT_ROOT,
    AE_LOG_DIR,
    MODEL_SAVE_DIR,
    ALL_PARAMETER_DIR,
    EVENT_PARAMETER_DIR,
    FIGURE_LOSS_DIR,
    RECON_FIGURE_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)


# =========================================================
# 3. Training configuration
# =========================================================
INPUT_SHAPE = (64, 512, 1)
BATCH_SIZE = 32
EPOCHS = 50
VALIDATION_RATIO = 0.2
RANDOM_STATE = 42
MAX_SAMPLES_PER_EVENT = 1500
PREDICT_BATCH_SIZE = 128
N_RECON_EXAMPLES = 8
LEARNING_RATE = 1e-4
CLIPNORM = 1.0
GLOBAL_NAME = "all_events"


# =========================================================
# 4. Helper functions
# =========================================================
def natural_event_key(event_name: str) -> int:
    match = re.search(r"event(\d+)", str(event_name))
    return int(match.group(1)) if match else 10**9


def extract_event_name_from_paths(files_array: np.ndarray, fallback_file: str) -> str:
    if len(files_array) > 0:
        text = str(files_array[0])
    else:
        text = str(fallback_file)

    match = re.search(r"event\d+", text)
    if match:
        return match.group(0)

    match = re.search(r"event\d+", str(fallback_file))
    if match:
        return match.group(0)

    return Path(fallback_file).stem


def clean_and_normalize_stft(
    x: np.ndarray,
    eps: float = 1e-8,
    clip_percentile: float = 99.5,
) -> np.ndarray:
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
    return x.astype(np.float32, copy=False)


def check_data_validity(x: np.ndarray, name: str = "x"):
    print(f"\n{name} data check:")
    print("shape:", x.shape)
    print("dtype:", x.dtype)
    print("min:", float(np.nanmin(x)))
    print("max:", float(np.nanmax(x)))
    print("mean:", float(np.nanmean(x)))
    print("std:", float(np.nanstd(x)))
    print("NaN count:", int(np.isnan(x).sum()))
    print("Inf count:", int(np.isinf(x).sum()))


def scan_all_event_metadata(
    train_file_glob: str,
    max_samples_per_event: int | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    train_files = sorted(glob.glob(train_file_glob))
    if not train_files:
        raise FileNotFoundError(f"No training files found: {train_file_glob}")

    rows = []
    start_idx = 0

    print(f"Found {len(train_files)} STFT files.")

    for train_file in train_files:
        with np.load(train_file, allow_pickle=True) as data:
            if "files" not in data:
                raise KeyError(f"'files' not found in {train_file}")
            file_paths = data["files"]

        event_name = extract_event_name_from_paths(file_paths, train_file)
        if max_samples_per_event is not None:
            file_paths = file_paths[:max_samples_per_event]

        n_samples = len(file_paths)
        for local_idx, wave_file in enumerate(file_paths):
            rows.append(
                {
                    "global_index": start_idx + local_idx,
                    "event": event_name,
                    "train_file": str(train_file),
                    "local_idx": int(local_idx),
                    "wave_file": str(wave_file),
                }
            )

        start_idx += n_samples

        print(f"{event_name}: scanned {n_samples} samples")

    metadata_df = pd.DataFrame(rows)
    metadata_df = metadata_df.sort_values(
        by=["event", "local_idx"],
        key=lambda s: s.map(natural_event_key) if s.name == "event" else s,
    ).reset_index(drop=True)
    metadata_df["global_index"] = np.arange(len(metadata_df), dtype=np.int64)
    event_slices = {
        event: group.index.to_numpy(dtype=np.int64)
        for event, group in metadata_df.groupby("event", sort=False)
    }

    print("\nAll-event metadata scanned.")
    print("Total samples:", len(metadata_df))
    return metadata_df, event_slices


def split_metadata_by_event(
    metadata_df: pd.DataFrame,
    validation_ratio: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    train_parts = []
    val_parts = []

    for _, event_df in metadata_df.groupby("event", sort=False):
        idx = event_df.index.to_numpy()
        perm = rng.permutation(idx)

        n_val = max(1, int(round(len(perm) * validation_ratio)))
        if n_val >= len(perm):
            n_val = max(1, len(perm) - 1)

        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        if len(train_idx) == 0:
            train_idx = perm[-1:]
            val_idx = perm[:-1]

        train_parts.append(metadata_df.loc[np.sort(train_idx)])
        val_parts.append(metadata_df.loc[np.sort(val_idx)])

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    return train_df, val_df


def load_event_subset_from_npz(
    train_file: str,
    local_indices: np.ndarray,
) -> np.ndarray:
    with np.load(train_file, allow_pickle=True) as data:
        if "stft" not in data:
            raise KeyError(f"'stft' not found in {train_file}")
        x = data["stft"][local_indices]

    x = clean_and_normalize_stft(x)
    x = np.expand_dims(x, axis=-1).astype(np.float32, copy=False)
    return x


class AllEventSTFTSequence(Sequence):
    def __init__(
        self,
        metadata_df: pd.DataFrame,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 87,
    ):
        self.metadata_df = metadata_df.reset_index(drop=True).copy()
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.event_batches = self._build_event_batches()
        self._cached_train_file = None
        self._cached_x = None
        self._cached_local_indices = None
        self.on_epoch_end()

    def _build_event_batches(self) -> list[dict]:
        event_batches = []
        for train_file, event_df in self.metadata_df.groupby("train_file", sort=False):
            local_indices = np.sort(event_df["local_idx"].to_numpy(dtype=np.int64))
            chunks = []
            for start in range(0, len(local_indices), self.batch_size):
                chunks.append(local_indices[start:start + self.batch_size])
            event_batches.append(
                {
                    "train_file": str(train_file),
                    "all_local_indices": local_indices,
                    "chunks": chunks,
                }
            )
        return event_batches

    def __len__(self) -> int:
        return len(self.batch_groups)

    def on_epoch_end(self):
        event_order = np.arange(len(self.event_batches))
        if self.shuffle:
            self.rng.shuffle(event_order)

        self.batch_groups = []
        for event_idx in event_order:
            event_group = self.event_batches[int(event_idx)]
            chunks = list(event_group["chunks"])
            if self.shuffle:
                self.rng.shuffle(chunks)
            for chunk in chunks:
                self.batch_groups.append((event_group["train_file"], chunk))

    def _load_cache_for_file(self, train_file: str):
        if self._cached_train_file == train_file:
            return

        event_group = next(item for item in self.event_batches if item["train_file"] == train_file)
        self._cached_x = load_event_subset_from_npz(
            train_file,
            event_group["all_local_indices"],
        )
        self._cached_local_indices = event_group["all_local_indices"]
        self._cached_train_file = train_file

    def __getitem__(self, index: int):
        train_file, local_indices = self.batch_groups[index]
        self._load_cache_for_file(train_file)

        positions = np.searchsorted(self._cached_local_indices, local_indices)
        x_batch = self._cached_x[positions]
        return x_batch, x_batch


def encode_metadata_in_batches(
    encoder,
    metadata_df: pd.DataFrame,
    batch_size: int = 128,
    output_dtype=np.float32,
) -> np.ndarray:
    features = []
    total = len(metadata_df)
    done = 0

    for train_file, event_df in metadata_df.groupby("train_file", sort=False):
        local_indices = event_df["local_idx"].to_numpy(dtype=np.int64)
        x_event = load_event_subset_from_npz(str(train_file), local_indices)

        for start in range(0, len(x_event), batch_size):
            end = min(start + batch_size, len(x_event))
            pred = encoder.predict(x_event[start:end], verbose=0)
            pred = pred.astype(output_dtype, copy=False)
            features.append(pred)
            done += len(pred)
            print(f"Encode: {done}/{total}")

        del x_event
        gc.collect()

    return np.concatenate(features, axis=0)


def save_loss_curve(history, output_csv: Path, output_png: Path):
    train_loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    loss_df = pd.DataFrame(
        {
            "Training Loss": train_loss,
            "Validation Loss": val_loss,
        }
    )

    loss_df.to_csv(output_csv, index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("All-event Autoencoder Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()

    print(f"Loss csv saved to: {output_csv}")
    print(f"Loss figure saved to: {output_png}")


def plot_reconstruction_example(
    x_input: np.ndarray,
    x_recon: np.ndarray,
    save_path: Path,
    title: str,
):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(np.squeeze(x_input), aspect="auto", cmap="viridis")
    plt.title("Input Spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(np.squeeze(x_recon), aspect="auto", cmap="viridis")
    plt.title("Reconstructed Spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_random_reconstruction_examples(
    autoencoder,
    metadata_df: pd.DataFrame,
    n_examples: int = 8,
):
    if len(metadata_df) == 0:
        return

    rng = np.random.default_rng(87)
    example_count = min(n_examples, len(metadata_df))
    example_indices = rng.choice(len(metadata_df), size=example_count, replace=False)

    for i, idx in enumerate(example_indices, start=1):
        row = metadata_df.iloc[int(idx)]
        x_input = load_event_subset_from_npz(
            str(row["train_file"]),
            np.asarray([int(row["local_idx"])], dtype=np.int64),
        )
        x_recon = autoencoder.predict(x_input, verbose=0)

        event_name = str(row["event"])
        save_path = RECON_FIGURE_DIR / f"{GLOBAL_NAME}_example_{i}_{event_name}.png"

        plot_reconstruction_example(
            x_input=x_input[0],
            x_recon=x_recon[0],
            save_path=save_path,
            title=f"All-event AE reconstruction example {i} | {event_name}",
        )

        print(f"Saved reconstruction example: {save_path}")
        del x_recon, x_input
        gc.collect()


def save_light_outputs(
    features_all: np.ndarray,
    file_all: np.ndarray,
    event_all: np.ndarray,
    event_slices: dict,
    loss_dict: dict,
) -> Path:
    global_output = {
        "features": features_all,
        "wave_file": file_all,
        "event": event_all,
        "loss": loss_dict,
    }

    global_npz_path = ALL_PARAMETER_DIR / f"{GLOBAL_NAME}_ACE_model_light.npz"
    np.savez(global_npz_path, **global_output)
    print(f"Saved global lightweight AE output to: {global_npz_path}")

    summary_rows = []
    for event_name, idx in event_slices.items():
        idx = np.asarray(idx)
        event_output = {
            "features": features_all[idx],
            "wave_file": file_all[idx],
            "event": event_all[idx],
            "loss": loss_dict,
        }

        event_npz_path = EVENT_PARAMETER_DIR / f"{event_name}_ACE_model_light.npz"
        np.savez(event_npz_path, **event_output)
        summary_rows.append(
            {
                "event": event_name,
                "sample_count": int(len(idx)),
                "npz_path": str(event_npz_path),
            }
        )
        print(f"Saved event-level lightweight AE output to: {event_npz_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(
        by="event",
        key=lambda s: s.map(natural_event_key),
    )

    summary_path = ALL_PARAMETER_DIR / f"{GLOBAL_NAME}_event_summary.xlsx"
    summary_df.to_excel(summary_path, index=False)
    print(f"Saved event summary to: {summary_path}")
    return global_npz_path


# =========================================================
# 5. Main workflow
# =========================================================
def main():
    current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
    print("Start time:", current_datetime)

    metadata_df, event_slices = scan_all_event_metadata(
        TRAIN_FILE_GLOB,
        max_samples_per_event=MAX_SAMPLES_PER_EVENT,
    )

    print("\nTotal samples for training:", len(metadata_df))

    train_meta, val_meta = split_metadata_by_event(
        metadata_df,
        validation_ratio=VALIDATION_RATIO,
        random_state=RANDOM_STATE,
    )

    print("Train sample count:", len(train_meta))
    print("Validation sample count:", len(val_meta))

    train_sequence = AllEventSTFTSequence(
        train_meta,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=RANDOM_STATE,
    )
    val_sequence = AllEventSTFTSequence(
        val_meta,
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=RANDOM_STATE,
    )

    preview_batch, _ = train_sequence[0]
    check_data_validity(preview_batch, name="train_batch_preview")
    del preview_batch
    gc.collect()

    encoder, decoder, autoencoder = Model_AE.CAE(
        input_shape=INPUT_SHAPE,
        filters=[8, 16, 32, 64, 128],
        encoding_filters=32,
        set_seed=[False, 87],
        summary=True,
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        clipnorm=CLIPNORM,
    )
    autoencoder.compile(optimizer=optimizer, loss="mse")

    csv_logger = tf.keras.callbacks.CSVLogger(
        str(AE_LOG_DIR / f"{GLOBAL_NAME}_pretrain_log.csv")
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=30,
        min_delta=1e-4,
        restore_best_weights=True,
    )
    checkpoint_path = MODEL_SAVE_DIR / f"{GLOBAL_NAME}_best_autoencoder.h5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1,
    )
    terminate_on_nan = tf.keras.callbacks.TerminateOnNaN()

    history = autoencoder.fit(
        train_sequence,
        validation_data=val_sequence,
        epochs=EPOCHS,
        callbacks=[
            csv_logger,
            early_stopping,
            checkpoint,
            reduce_lr,
            terminate_on_nan,
        ],
        workers=0,
        use_multiprocessing=False,
        max_queue_size=2,
    )

    if checkpoint_path.exists():
        autoencoder.load_weights(str(checkpoint_path))
        print(f"Loaded best autoencoder weights from: {checkpoint_path}")

    save_loss_curve(
        history,
        output_csv=FIGURE_LOSS_DIR / f"{GLOBAL_NAME}_Loss.csv",
        output_png=FIGURE_LOSS_DIR / f"{GLOBAL_NAME}_Loss.png",
    )

    encoder_weight_path = MODEL_SAVE_DIR / f"{GLOBAL_NAME}_Pretrain_encoder.h5"
    autoencoder_weight_path = MODEL_SAVE_DIR / f"{GLOBAL_NAME}_Pretrain_autoencoder.h5"
    decoder_weight_path = MODEL_SAVE_DIR / f"{GLOBAL_NAME}_Pretrain_decoder.h5"

    encoder.save_weights(str(encoder_weight_path))
    autoencoder.save_weights(str(autoencoder_weight_path))
    decoder.save_weights(str(decoder_weight_path))

    print("\nSaved global pretrained weights:")
    print("Encoder:", encoder_weight_path)
    print("Autoencoder:", autoencoder_weight_path)
    print("Decoder:", decoder_weight_path)

    print("\nExtracting all-event latent features...")
    features_all = encode_metadata_in_batches(
        encoder,
        metadata_df,
        batch_size=PREDICT_BATCH_SIZE,
        output_dtype=np.float32,
    )
    check_data_validity(features_all, name="features_all")
    print("features_all shape:", features_all.shape)

    print("\nPlotting random reconstruction examples only...")
    plot_random_reconstruction_examples(
        autoencoder=autoencoder,
        metadata_df=metadata_df,
        n_examples=N_RECON_EXAMPLES,
    )

    loss_dict = {
        "train_loss": history.history.get("loss", []),
        "val_loss": history.history.get("val_loss", []),
    }

    global_npz_path = save_light_outputs(
        features_all=features_all,
        file_all=metadata_df["wave_file"].to_numpy(),
        event_all=metadata_df["event"].to_numpy(),
        event_slices=event_slices,
        loss_dict=loss_dict,
    )

    del train_sequence, val_sequence, train_meta, val_meta, metadata_df
    gc.collect()
    tf.keras.backend.clear_session()

    print("\nDone.")
    print(f"Global weights saved to: {MODEL_SAVE_DIR}")
    print(f"Global lightweight output saved to: {global_npz_path}")
    print(f"Event-level lightweight outputs saved to: {EVENT_PARAMETER_DIR}")


if __name__ == "__main__":
    main()
