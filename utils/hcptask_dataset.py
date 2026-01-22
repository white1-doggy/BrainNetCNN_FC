import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class HCPTaskDataset(Dataset):
    """Dataset for per-task binary FC data organized by subject/task/label."""

    def __init__(
        self,
        subject_list,
        task_config,
        fc_root,
        roi_ids=None,
        task_name=None,
    ):
        self.subject_list = list(subject_list)
        self.task_config = self._load_task_config(task_config) if task_config else {}
        self.fc_root = fc_root
        self.roi_ids = self._resolve_roi_ids(roi_ids)
        self.task_name = task_name
        if not self.task_name:
            raise ValueError("HCPTaskDataset requires a single task name.")
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
        for subject in tqdm(os.listdir(self.fc_root), desc="Indexing HCPTask FC data"):
            if subject_set and subject not in subject_set:
                continue
            subject_dir = os.path.join(self.fc_root, subject)
            if not os.path.isdir(subject_dir):
                continue
            task_dir = os.path.join(subject_dir, self.task_name)
            if not os.path.isdir(task_dir):
                continue
            for label in os.listdir(task_dir):
                label_dir = os.path.join(task_dir, label)
                if not os.path.isdir(label_dir):
                    continue
                for fc_name in sorted(os.listdir(label_dir)):
                    fc_path = os.path.join(label_dir, fc_name)
                    if not os.path.isfile(fc_path):
                        continue
                    if not fc_name.endswith((".pt", ".npy", ".txt")) and "." in fc_name:
                        continue
                    samples.append(
                        {
                            "task": self.task_name,
                            "subject": subject,
                            "label": label,
                            "fc_path": fc_path,
                        }
                    )
        return samples

    def _load_fc(self, path):
        if path.endswith(".pt") or "." not in os.path.basename(path):
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

    def _resolve_label(self, label_value):
        rules = self.task_config.get("task_label_rules", {}).get(self.task_name, {})
        if "label_map" in rules and str(label_value) in rules["label_map"]:
            return int(rules["label_map"][str(label_value)])
        if "label_to_id" in rules and str(label_value) in rules["label_to_id"]:
            return int(rules["label_to_id"][str(label_value)])
        if "labels" in rules and label_value in rules["labels"]:
            return int(rules["labels"].index(label_value))
        if isinstance(label_value, str) and label_value.isdigit():
            return int(label_value)
        raise ValueError(f"Unrecognized label mapping for task {self.task_name}: {label_value}")

    def _infer_num_rois(self):
        if not self.samples:
            return 0
        sample_fc = self._load_fc(self.samples[0]["fc_path"])
        if self.roi_ids:
            sample_fc = self._maybe_select_rois(sample_fc)
        return int(sample_fc.shape[0])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        fc = self._load_fc(item["fc_path"])
        fc = self._maybe_select_rois(fc)
        label = self._resolve_label(item["label"])
        return fc.unsqueeze(0), torch.tensor(label, dtype=torch.long)
