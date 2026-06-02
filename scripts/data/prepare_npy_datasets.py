import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import io
from scipy.io import arff

from generate_sines_dataset import generate_sines


DROP_COLUMN_NAMES = {"date", "datetime", "timestamp", "time"}


def read_csv(csv_path):
    try:
        return pd.read_csv(csv_path, header=0)
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, header=0, encoding="latin1")


def save_csv_as_npy(csv_path, npy_path):
    data = read_csv(csv_path)
    drop_columns = [
        column
        for column in data.columns
        if str(column).strip().lower() in DROP_COLUMN_NAMES
        or str(column).strip().lower().startswith("unnamed")
    ]
    if drop_columns:
        data = data.drop(columns=drop_columns)

    numeric = data.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    numeric = numeric.interpolate(limit_direction="both").ffill().bfill()
    numeric = numeric.fillna(numeric.mean(numeric_only=True)).fillna(0.0)

    values = numeric.to_numpy(dtype=np.float32)
    np.save(npy_path, values)
    print(f"saved {npy_path} shape={values.shape}")


def save_mat_ts_as_npy(mat_path, npy_path):
    data = io.loadmat(mat_path)
    if "ts" not in data:
        raise KeyError(f"Expected key `ts` in {mat_path}. Available keys: {sorted(data)}")
    values = np.asarray(data["ts"], dtype=np.float32)
    np.save(npy_path, values)
    print(f"saved {npy_path} shape={values.shape}")


def save_eeg_eye_arff_as_npy(arff_path, npy_path, label_path):
    data, _ = arff.loadarff(arff_path)
    frame = pd.DataFrame(data)
    if "eyeDetection" not in frame.columns:
        raise KeyError(f"Expected label column `eyeDetection` in {arff_path}.")

    labels = frame.pop("eyeDetection").astype(np.int64).to_numpy()
    values = frame.to_numpy(dtype=np.float32)
    np.save(npy_path, values)
    np.save(label_path, labels)
    print(f"saved {npy_path} shape={values.shape}")
    print(f"saved {label_path} shape={labels.shape}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert benchmark datasets to unified .npy files.")
    parser.add_argument("--dataset-dir", default="data/datasets")
    parser.add_argument("--sine-series-length", type=int, default=1024)
    parser.add_argument("--sine-channels", type=int, default=5)
    parser.add_argument("--sine-num-series", type=int, default=1000)
    parser.add_argument("--sine-seed", type=int, default=123)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    csv_jobs = [
        (dataset_dir / "ETTh1.csv", dataset_dir / "etth1.npy"),
        (dataset_dir / "ETTh2.csv", dataset_dir / "etth2.npy"),
        (dataset_dir / "ETTm1.csv", dataset_dir / "ettm1.npy"),
        (dataset_dir / "ETTm2.csv", dataset_dir / "ettm2.npy"),
        (dataset_dir / "AirQuality_clean.csv", dataset_dir / "air.npy"),
        (dataset_dir / "stock_data.csv", dataset_dir / "stocks.npy"),
        (dataset_dir / "energy_data.csv", dataset_dir / "energy.npy"),
        (dataset_dir / "electricity.csv", dataset_dir / "electricity.npy"),
        (dataset_dir / "exchange_rate.csv", dataset_dir / "exchange.npy"),
        (dataset_dir / "weather.csv", dataset_dir / "weather.npy"),
        (dataset_dir / "BE.csv", dataset_dir / "be.npy"),
        (dataset_dir / "DE.csv", dataset_dir / "de.npy"),
        (dataset_dir / "FR.csv", dataset_dir / "fr.npy"),
    ]
    if not (dataset_dir / "ETTh1.csv").exists() and (dataset_dir / "ETTh.csv").exists():
        csv_jobs.append((dataset_dir / "ETTh.csv", dataset_dir / "etth1.npy"))

    for csv_path, npy_path in csv_jobs:
        if csv_path.exists():
            save_csv_as_npy(csv_path, npy_path)
        elif npy_path.exists():
            print(f"skip {csv_path}; existing {npy_path}")

    legacy_etth_path = dataset_dir / "etth.npy"
    if legacy_etth_path.exists() and (dataset_dir / "etth1.npy").exists():
        legacy_etth_path.unlink()
        print(f"removed legacy {legacy_etth_path}; use {dataset_dir / 'etth1.npy'}")

    sines = generate_sines(args.sine_num_series, args.sine_series_length, args.sine_channels, args.sine_seed)
    np.save(dataset_dir / "sines.npy", sines)
    print(f"saved {dataset_dir / 'sines.npy'} shape={sines.shape} seed={args.sine_seed}")

    fmri_dir = dataset_dir / "fMRI"
    if fmri_dir.exists():
        for mat_path in sorted(fmri_dir.glob("sim*.mat")):
            save_mat_ts_as_npy(mat_path, dataset_dir / f"fmri_{mat_path.stem}.npy")
        sim4_path = fmri_dir / "sim4.mat"
        if sim4_path.exists():
            save_mat_ts_as_npy(sim4_path, dataset_dir / "fmri.npy")
    elif (dataset_dir / "fmri.npy").exists():
        print(f"skip {fmri_dir}; existing {dataset_dir / 'fmri.npy'}")

    eeg_eye_path = dataset_dir / "EEG_Eye_State.arff"
    if eeg_eye_path.exists():
        save_eeg_eye_arff_as_npy(
            eeg_eye_path,
            dataset_dir / "eeg.npy",
            dataset_dir / "eeg_label.npy",
        )
    elif (dataset_dir / "eeg.npy").exists():
        print(f"skip {eeg_eye_path}; existing {dataset_dir / 'eeg.npy'}")


if __name__ == "__main__":
    main()

