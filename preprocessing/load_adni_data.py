import json
import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _normalize_subject_token(subject):
    subject = str(subject).strip()
    return subject[:-4] if subject.endswith('.npy') else subject


def _match_file_by_subject(subject, all_files):
    exact = f'{subject}.npy'
    if exact in all_files:
        return exact

    prefix = f'{subject}_'
    hits = [f for f in all_files if f.startswith(prefix)]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        hits.sort()
        return hits[0]
    return None


def _balance_binary(files, labels, class_a, class_b):
    files = np.array(files)
    labels = np.array(labels)

    idx_a = np.where(labels == class_a)[0]
    idx_b = np.where(labels == class_b)[0]
    n = min(len(idx_a), len(idx_b))
    if n == 0:
        raise ValueError(f'Cannot balance: {class_a}={len(idx_a)}, {class_b}={len(idx_b)}')

    rng = np.random.RandomState(1234)
    keep_a = rng.choice(idx_a, size=n, replace=False)
    keep_b = rng.choice(idx_b, size=n, replace=False)
    keep = np.concatenate([keep_a, keep_b])
    rng.shuffle(keep)

    return files[keep].tolist(), labels[keep].tolist()


def _load_or_create_split(samples, labels, split_txt_path, test_size):
    if split_txt_path and os.path.isfile(split_txt_path):
        with open(split_txt_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        if all(k in loaded for k in ['train', 'test']):
            return loaded, split_txt_path
        if '0' in loaded and all(k in loaded['0'] for k in ['train', 'test']):
            return loaded['0'], split_txt_path
        raise ValueError(f'Invalid split file format: {split_txt_path}')

    samples = np.array(samples)
    labels = np.array(labels)

    train_x, test_x, train_y, test_y = train_test_split(
        samples, labels, test_size=test_size, random_state=1234, stratify=labels
    )

    split = dict(
        train=train_x.tolist(),
        test=test_x.tolist(),
    )

    out_path = split_txt_path or os.path.join(os.getcwd(), 'split.txt')
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(split, f, indent=2)

    return split, out_path


def main(args):
    signal_dir = args['adni_signal_dir']

    csv_path = args['adni_label_csv']
    label_df = pd.read_csv(csv_path)
    if not {'subject', 'label'}.issubset(set(label_df.columns)):
        raise ValueError('adni_label_csv must include columns: subject, label')

    if args.get('adni_classes'):
        chosen_classes = list(args['adni_classes'])
    else:
        classes_from_csv = sorted({str(x).strip() for x in label_df['label'].dropna().tolist()})
        if len(classes_from_csv) != 2:
            raise ValueError(f'CSV label has {len(classes_from_csv)} classes. '
                             f'Please provide --adni_classes for binary task selection.')
        chosen_classes = classes_from_csv

    class_a, class_b = chosen_classes

    all_npy_files = [f for f in os.listdir(signal_dir) if f.endswith('.npy')]
    all_npy_files.sort()

    matched_files, matched_labels = [], []
    missing_subjects = []

    for _, row in label_df.iterrows():
        subject = _normalize_subject_token(row['subject'])
        label = str(row['label']).strip()
        if label not in chosen_classes:
            continue

        matched = _match_file_by_subject(subject, all_npy_files)
        if matched is None:
            missing_subjects.append(subject)
            continue

        matched_files.append(os.path.join(signal_dir, matched))
        matched_labels.append(label)

    if missing_subjects:
        print(f'\nWarning: {len(missing_subjects)} subjects from csv were not matched to npy files.')

    if not matched_files:
        raise ValueError('No ADNI files matched from csv subject/label mapping for selected classes')

    print('\nSelected class counts before balancing:')
    before_counts = Counter(matched_labels)
    for klass in chosen_classes:
        print(f'  {klass}: {before_counts.get(klass, 0)}')

    balanced_files, balanced_labels = _balance_binary(matched_files, matched_labels, class_a, class_b)

    print('\nSelected class counts after balancing:')
    after_counts = Counter(balanced_labels)
    for klass in chosen_classes:
        print(f'  {klass}: {after_counts.get(klass, 0)}')

    split, used_split_txt_path = _load_or_create_split(
        samples=balanced_files,
        labels=balanced_labels,
        split_txt_path=args.get('split_txt_path'),
        test_size=args.get('test_size', 0.2),
    )

    print(f'\nUsing split file: {used_split_txt_path}')

    label_to_int = {class_a: 0, class_b: 1}
    int_to_label = {0: class_a, 1: class_b}

    return dict(
        adni_files=balanced_files,
        adni_labels=balanced_labels,
        label_to_int=label_to_int,
        int_to_label=int_to_label,
        adni_fold_splits={'0': dict(train=split['train'], val=[], test=split['test'])},
        split_txt_path=used_split_txt_path,
        outcome_names=chosen_classes,
        n_folds=1,
        start_fold=0,
        end_fold=1,
    )


if __name__ == '__main__':
    raise SystemExit('This module is intended to be imported by parse_training_args.py')
