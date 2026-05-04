from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class BiosignalDataset(Dataset):
    """Dataset for multimodal EEG, GSR, and PPG binary classification.

    Expected directory structure:

        DATA_SLICED/
        ├── EEG_sliced/
        │   ├── 0/
        │   └── 1/
        ├── GSR_sliced/
        │   ├── 0/
        │   └── 1/
        └── PPG_sliced/
            ├── 0/
            └── 1/

    Each modality is expected to contain matching .npy filenames under the
    same class-label folder.
    """

    VALID_SPLITS = {"train", "val", "test"}
    MODALITY_DIRS = {
        "eeg": "EEG_sliced",
        "gsr": "GSR_sliced",
        "ppg": "PPG_sliced",
    }

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        split_path: Optional[str | Path] = None,
        load_into_memory: bool = True,
        verbose: bool = True,
    ) -> None:
        if split not in self.VALID_SPLITS:
            raise ValueError(f"split must be one of {sorted(self.VALID_SPLITS)}, got {split!r}")

        self.data_dir = Path(data_dir)
        self.split = split
        self.split_path = Path(split_path) if split_path is not None else None
        self.load_into_memory = load_into_memory
        self.verbose = verbose

        self._validate_data_dirs()

        self.split_keys = self._load_split_keys()
        self.index = self._build_index()
        self.samples = [self._load_sample(item) for item in self.index] if load_into_memory else None

        if self.verbose:
            cache_text = "loaded into memory" if load_into_memory else "indexed"
            print(f"[INFO] {split}: {len(self.index)} samples {cache_text} ✅")

    def _validate_data_dirs(self) -> None:
        if not self.data_dir.exists():
            raise FileNotFoundError(f"data_dir does not exist: {self.data_dir}")

        for dirname in self.MODALITY_DIRS.values():
            path = self.data_dir / dirname
            if not path.exists():
                raise FileNotFoundError(f"Missing modality directory: {path}")

    def _find_split_file(self) -> Optional[Path]:
        if self.split_path is None:
            return None

        if self.split_path.is_file():
            return self.split_path

        candidates = [
            f"{self.split}.txt",
            f"{self.split}.csv",
            f"{self.split}.json",
            f"{self.split}.npy",
            f"{self.split}_subjects.txt",
            f"{self.split}_subjects.csv",
            f"{self.split}_files.txt",
            f"{self.split}_files.csv",
            f"subject_{self.split}.txt",
            f"subject_{self.split}.csv",
        ]

        for filename in candidates:
            path = self.split_path / filename
            if path.exists():
                return path

        raise FileNotFoundError(
            f"No split file found for split={self.split!r} under {self.split_path}. "
            "Expected files like train.txt, val.txt, test.txt, or train_subjects.txt."
        )

    def _load_split_keys(self) -> Optional[set[str]]:
        split_file = self._find_split_file()
        if split_file is None:
            return None

        suffix = split_file.suffix.lower()

        if suffix == ".npy":
            raw_items = np.load(split_file, allow_pickle=True).tolist()
            if not isinstance(raw_items, list):
                raw_items = [raw_items]

        elif suffix == ".json":
            with split_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            raw_items = data.get(self.split, data) if isinstance(data, dict) else data

        elif suffix in {".txt", ".csv"}:
            raw_items = []
            with split_file.open("r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    for token in row:
                        token = token.strip()
                        if token:
                            raw_items.append(token)

        else:
            raise ValueError(f"Unsupported split file format: {split_file}")

        keys: set[str] = set()
        ignored = {"file", "filename", "path", "subject", "subject_id", "split", "label"}

        for item in raw_items:
            text = str(item).strip()
            if not text or text.lower() in ignored:
                continue

            path = Path(text)
            keys.add(text)
            keys.add(path.name)
            keys.add(path.stem)

        return keys

    def _belongs_to_split(self, filename: str) -> bool:
        if self.split_keys is None:
            return True

        base = Path(filename).name
        stem = Path(filename).stem

        if base in self.split_keys or stem in self.split_keys:
            return True

        return any(len(key) >= 3 and (stem.startswith(key) or key in stem) for key in self.split_keys)

    def _build_index(self) -> list[tuple[Path, Path, Path, float, int | None]]:
        items: list[tuple[Path, Path, Path, float, int | None]] = []

        for label_name in ("0", "1"):
            label = float(label_name)

            eeg_dir = self.data_dir / self.MODALITY_DIRS["eeg"] / label_name
            gsr_dir = self.data_dir / self.MODALITY_DIRS["gsr"] / label_name
            ppg_dir = self.data_dir / self.MODALITY_DIRS["ppg"] / label_name

            if not eeg_dir.exists():
                continue

            filenames = sorted(
                path.name for path in eeg_dir.glob("*.npy") if self._belongs_to_split(path.name)
            )

            for filename in filenames:
                eeg_path = eeg_dir / filename
                gsr_path = gsr_dir / filename
                ppg_path = ppg_dir / filename

                if not gsr_path.exists() or not ppg_path.exists():
                    if self.verbose:
                        print(f"[WARN] Missing paired modality file: {filename}")
                    continue

                eeg = np.load(eeg_path, mmap_mode="r")
                gsr = np.load(gsr_path, mmap_mode="r")
                ppg = np.load(ppg_path, mmap_mode="r")

                if self._is_stacked_window_file(eeg, gsr, ppg):
                    for window_idx in range(eeg.shape[0]):
                        items.append((eeg_path, gsr_path, ppg_path, label, window_idx))
                else:
                    items.append((eeg_path, gsr_path, ppg_path, label, None))

        if not items:
            raise RuntimeError(
                f"No samples found for split={self.split!r}. "
                "Check data_dir, split files, and matching .npy filenames."
            )

        return items

    @staticmethod
    def _is_stacked_window_file(eeg: np.ndarray, gsr: np.ndarray, ppg: np.ndarray) -> bool:
        return (
            eeg.ndim >= 3
            and gsr.ndim >= 2
            and ppg.ndim >= 2
            and eeg.shape[0] == gsr.shape[0] == ppg.shape[0]
            and eeg.shape[0] > 1
        )

    @staticmethod
    def _to_float32(array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, dtype=torch.float32)

    def _load_sample(self, item):
        eeg_path, gsr_path, ppg_path, label, window_idx = item

        eeg = np.load(eeg_path)
        gsr = np.load(gsr_path)
        ppg = np.load(ppg_path)

        if window_idx is not None:
            eeg = eeg[window_idx]
            gsr = gsr[window_idx]
            ppg = ppg[window_idx]

        return (
            eeg.astype(np.float32, copy=False),
            gsr.astype(np.float32, copy=False),
            ppg.astype(np.float32, copy=False),
            np.float32(label),
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        if self.samples is not None:
            eeg, gsr, ppg, label = self.samples[idx]
        else:
            eeg, gsr, ppg, label = self._load_sample(self.index[idx])

        return (
            self._to_float32(eeg),
            self._to_float32(gsr),
            self._to_float32(ppg),
            torch.tensor(label, dtype=torch.float32),
        )
