import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class HCP7TaskDataset(Dataset):
    """Dataset for precomputed FC matrices organized by subject/task/run."""

    def __init__(
        self,
        subject_list,
        task_config,
        fc_root,
        roi_ids=None,
    ):
        self.subject_list = list(subject_list)
        self.task_config = self._load_task_config(task_config)
        self.task_name_list = self.task_config["task_name_list"]
        self.fc_root = fc_root
        self.roi_ids = self._resolve_roi_ids(roi_ids)
        self.samples = self._build_fc_index()
        self.num_rois = self._infer_num_rois()

    def _load_task_config(self, task_config):
        if isinstance(task_config, str):
            with open(task_config, "r", encoding="utf-8") as f:
                return json.load(f)
        return task_config

    def _resolve_roi_ids(self, roi_ids):
        if roi_ids is None:
            return None
        if isinstance(roi_ids, str):
            return [int(v) for v in roi_ids.split(",") if v.strip()]
        if isinstance(roi_ids, (list, tuple, np.ndarray)):
            return [int(v) for v in roi_ids]
        return roi_ids

    def _build_fc_index(self):
        samples = []
        subject_set = set(self.subject_list)
        for subject in os.listdir(self.fc_root):
            if subject_set and subject not in subject_set:
                continue
            subject_dir = os.path.join(self.fc_root, subject)
            if not os.path.isdir(subject_dir):
                continue
            for task in self.task_name_list:
                task_dir = os.path.join(subject_dir, task)
                if not os.path.isdir(task_dir):
                    continue
                for run in os.listdir(task_dir):
                    run_dir = os.path.join(task_dir, run)
                    if not os.path.isdir(run_dir):
                        continue
                    label_dirs = [
                        entry
                        for entry in os.listdir(run_dir)
                        if os.path.isdir(os.path.join(run_dir, entry))
                    ]
                    if label_dirs:
                        for label in label_dirs:
                            label_dir = os.path.join(run_dir, label)
                            fc_files = sorted(
                                f for f in os.listdir(label_dir) if f.endswith((".pt", ".npy", ".txt"))
                            )
                            for fc_name in fc_files:
                                samples.append(
                                    {
                                        "task": task,
                                        "subject": subject,
                                        "run": run,
                                        "label": label,
                                        "fc_path": os.path.join(label_dir, fc_name),
                                    }
                                )
                    else:
                        fc_files = sorted(
                            f for f in os.listdir(run_dir) if f.endswith((".pt", ".npy", ".txt"))
                        )
                        for fc_name in fc_files:
                            samples.append(
                                {
                                    "task": task,
                                    "subject": subject,
                                    "run": run,
                                    "label": None,
                                    "fc_path": os.path.join(run_dir, fc_name),
                                }
                            )
        return samples

    def _load_fc(self, path):
        if path.endswith(".pt"):
            fc = torch.load(path)
        elif path.endswith(".npy"):
            fc = torch.from_numpy(np.load(path))
        else:
            fc = torch.from_numpy(np.loadtxt(path))
        return fc.float()

    def _maybe_select_rois(self, fc):
        if not self.roi_ids:
            return fc
        roi_idx = torch.tensor(self.roi_ids, dtype=torch.long)
        return fc.index_select(0, roi_idx).index_select(1, roi_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        fc = self._load_fc(item["fc_path"])
        fc = self._maybe_select_rois(fc)
        label = self.task_config["task_name_to_id"][item["task"]]
        return fc.unsqueeze(0), torch.tensor(label, dtype=torch.long)

    def _infer_num_rois(self):
        if not self.samples:
            return 0
        sample_fc = self._load_fc(self.samples[0]["fc_path"])
        if self.roi_ids:
            sample_fc = self._maybe_select_rois(sample_fc)
        return int(sample_fc.shape[0])
