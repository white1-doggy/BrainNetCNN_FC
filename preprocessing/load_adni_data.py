import json
import os
from collections import Counter

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


def _parse_label(filename):
    return filename.split('_')[0]


def _load_or_create_splits(samples, labels, n_folds, split_txt_path):
    if split_txt_path and os.path.isfile(split_txt_path):
        with open(split_txt_path, 'r', encoding='utf-8') as f:
            return json.load(f), split_txt_path

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1234)
    labels = np.array(labels)
    samples = np.array(samples)

    fold_splits = {}
    for fold, (trainval_idx, test_idx) in enumerate(skf.split(samples, labels)):
        trainval_labels = labels[trainval_idx]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1234 + fold)
        train_rel, val_rel = next(sss.split(trainval_idx, trainval_labels))

        train_idx = trainval_idx[train_rel]
        val_idx = trainval_idx[val_rel]

        fold_splits[str(fold)] = dict(
            train=samples[train_idx].tolist(),
            val=samples[val_idx].tolist(),
            test=samples[test_idx].tolist(),
        )

    out_path = split_txt_path or os.path.join(os.getcwd(), 'adni_splits.txt')
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(fold_splits, f, indent=2)

    return fold_splits, out_path


def main(args):
    signal_dir = args['adni_signal_dir']
    chosen_classes = args['adni_classes']
    n_folds = args['n_folds']

    npy_files = [f for f in os.listdir(signal_dir) if f.endswith('.npy')]
    npy_files.sort()

    labels = [_parse_label(f) for f in npy_files]
    class_counts = Counter(labels)
    print('\nADNI class counts (all files):')
    for klass, count in sorted(class_counts.items()):
        print(f'  {klass}: {count}')

    selected = [(f, y) for f, y in zip(npy_files, labels) if y in chosen_classes]
    if not selected:
        raise ValueError(f'No files found for selected classes: {chosen_classes}')

    selected_files = [os.path.join(signal_dir, f) for f, _ in selected]
    selected_labels = [y for _, y in selected]

    selected_counts = Counter(selected_labels)
    print('\nSelected binary task class counts:')
    for klass in chosen_classes:
        print(f'  {klass}: {selected_counts.get(klass, 0)}')

    label_to_int = {chosen_classes[0]: 0, chosen_classes[1]: 1}
    int_to_label = {v: k for k, v in label_to_int.items()}

    fold_splits, used_split_txt_path = _load_or_create_splits(
        samples=selected_files,
        labels=selected_labels,
        n_folds=n_folds,
        split_txt_path=args.get('split_txt_path')
    )

    print(f'\nUsing split file: {used_split_txt_path}')

    return dict(
        adni_files=selected_files,
        adni_labels=selected_labels,
        label_to_int=label_to_int,
        int_to_label=int_to_label,
        adni_fold_splits=fold_splits,
        split_txt_path=used_split_txt_path,
        outcome_names=chosen_classes,
    )


if __name__ == '__main__':
    raise SystemExit('This module is intended to be imported by parse_training_args.py')
