import datetime
import json
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from sklearn import neural_network, svm
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.hcp7task_dataset import HCP7TaskDataset
from utils.util_args import models_dir, output_dir, performance_dir, seed
from utils.util_classes import ClassFromDict, PervaizBNCNN, HeSexBNCNN, He58behaviorsBNCNN, KawaharaBNCNN
from utils.util_funcs import ensure_subfolder_in_folder


def _load_subject_splits(subject_list_path):
    train_subjects = []
    val_subjects = []
    test_subjects = []
    current_split = None

    with open(subject_list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key = line.lower()
            if key == "train_subjects":
                current_split = "train"
                continue
            if key == "val_subjects":
                current_split = "val"
                continue
            if key == "test_subjects":
                current_split = "test"
                continue

            if current_split == "train":
                train_subjects.append(line)
            elif current_split == "val":
                val_subjects.append(line)
            elif current_split == "test":
                test_subjects.append(line)
            else:
                raise ValueError(f"Subject found before split header: {line}")

    if not train_subjects or not val_subjects or not test_subjects:
        raise ValueError("Subject list must include train_subjects/val_subjects/test_subjects")

    return train_subjects, val_subjects, test_subjects


def _load_task_config(task_config_path):
    with open(task_config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_datasets(params):
    task_config = _load_task_config(params.task_config)
    train_subjects, val_subjects, test_subjects = _load_subject_splits(params.subject_list)
    roi_ids = params.roi_ids if getattr(params, "roi_ids", None) else None
    task_list = [task for task in params.tasks if task] if getattr(params, "tasks", None) else None
    label_from_dir = getattr(params, "label_from_dir", False)

    train_dataset = HCP7TaskDataset(
        subject_list=train_subjects,
        task_config=task_config,
        fc_root=params.fc_root,
        roi_ids=roi_ids,
        task_list=task_list,
        label_from_dir=label_from_dir,
    )
    val_dataset = HCP7TaskDataset(
        subject_list=val_subjects,
        task_config=task_config,
        fc_root=params.fc_root,
        roi_ids=roi_ids,
        task_list=task_list,
        label_from_dir=label_from_dir,
    )
    test_dataset = HCP7TaskDataset(
        subject_list=test_subjects,
        task_config=task_config,
        fc_root=params.fc_root,
        roi_ids=roi_ids,
        task_list=task_list,
        label_from_dir=label_from_dir,
    )

    if label_from_dir:
        train_labels = [label.item() for _, label in train_dataset]
        num_classes = int(max(train_labels)) + 1 if train_labels else 0
    else:
        num_classes = len(task_config["task_name_list"])
    return train_dataset, val_dataset, test_dataset, num_classes


def _build_flat_features(dataset, desc=None):
    if len(dataset) == 0:
        return np.empty((0, 0)), np.empty((0,))
    sample_fc, _ = dataset[0]
    n = sample_fc.shape[-1]
    tri_idx = np.triu_indices(n, k=1)
    features = []
    labels = []
    data_iter = tqdm(dataset, desc=desc) if desc else dataset
    for fc, label in data_iter:
        fc_np = fc.squeeze(0).numpy()
        features.append(fc_np[tri_idx])
        labels.append(label.item())
    return np.stack(features, axis=0), np.array(labels)


def _build_bncnn(architecture, example, num_classes):
    dummy = ClassFromDict(
        dict(
            n_classes=num_classes,
            multiclass=True,
            n_outcomes=1,
        )
    )
    switcher = dict(
        kawahara=KawaharaBNCNN,
        he_sex=HeSexBNCNN,
        pervaiz=PervaizBNCNN,
        he_58=He58behaviorsBNCNN,
    )
    net_cls = switcher.get(architecture, PervaizBNCNN)
    return net_cls(example, dummy)


def _compute_class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def _predict_probabilities(outputs, num_classes):
    if num_classes > 2:
        return torch.softmax(outputs, dim=1).cpu().numpy()
    return outputs.detach().cpu().numpy()


def _compute_epoch_metrics(y_true, y_pred, y_prob, num_classes):
    metrics = {
        "acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary" if num_classes == 2 else "macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary" if num_classes == 2 else "macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="binary" if num_classes == 2 else "macro", zero_division=0),
    }
    try:
        if num_classes == 2:
            metrics["auc"] = roc_auc_score(y_true, y_prob[:, 1])
        else:
            metrics["auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr")
    except ValueError:
        metrics["auc"] = float("nan")
    return metrics


def _format_label_distribution(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes)
    return ", ".join([f"{idx}:{count}" for idx, count in enumerate(counts)])


def _sklearn_predict_proba(estimator, X, num_classes):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)
    scores = estimator.decision_function(X)
    if num_classes == 2:
        scores = scores.reshape(-1, 1)
        prob_pos = 1 / (1 + np.exp(-scores))
        return np.concatenate([1 - prob_pos, prob_pos], axis=1)
    scores = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def _print_metrics(prefix, metrics):
    print(
        f"{prefix} - "
        f"acc: {metrics['acc']:.4f}, "
        f"auc: {metrics['auc']:.4f}, "
        f"precision: {metrics['precision']:.4f}, "
        f"recall: {metrics['recall']:.4f}, "
        f"f1: {metrics['f1']:.4f}"
    )


def _train_bncnn(params):
    train_dataset, val_dataset, test_dataset, num_classes = _build_datasets(params)
    example, _ = train_dataset[0]
    example = example.unsqueeze(0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = _build_bncnn(params.architecture, example, num_classes).to(device)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    if params.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.wd)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=params.lr, momentum=params.momentum, weight_decay=params.wd)

    class_weights = _compute_class_weights(
        [label.item() for _, label in train_dataset], num_classes
    ).to(device)
    if num_classes > 2:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.BCELoss(weight=class_weights)

    best_state = None
    best_val_loss = float("inf")
    history = []

    for epoch in range(params.n_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch} [train]") if params.verbose else train_loader
        for inputs, labels in train_iter:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if num_classes > 2:
                loss = criterion(outputs, labels)
                preds = outputs.argmax(dim=1)
            else:
                targets = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
                loss = criterion(outputs, targets)
                preds = outputs.argmax(dim=1)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_correct += (preds == labels).sum().item()

        train_loss /= len(train_dataset)
        train_acc = train_correct / len(train_dataset)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            val_iter = tqdm(val_loader, desc=f"Epoch {epoch} [val]") if params.verbose else val_loader
            for inputs, labels in val_iter:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                if num_classes > 2:
                    loss = criterion(outputs, labels)
                    preds = outputs.argmax(dim=1)
                else:
                    targets = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
                    loss = criterion(outputs, targets)
                    preds = outputs.argmax(dim=1)
                val_loss += loss.item() * inputs.size(0)
                val_correct += (preds == labels).sum().item()

        val_loss /= len(val_dataset)
        val_acc = val_correct / len(val_dataset)
        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss, train_acc=train_acc, val_acc=val_acc))

        test_true = []
        test_pred = []
        test_prob = []
        with torch.no_grad():
            test_iter = tqdm(test_loader, desc=f"Epoch {epoch} [test]") if params.verbose else test_loader
            for inputs, labels in test_iter:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                probs = _predict_probabilities(outputs, num_classes)
                preds = outputs.argmax(dim=1).cpu().numpy()
                test_true.append(labels.cpu().numpy())
                test_pred.append(preds)
                test_prob.append(probs)

        test_true = np.concatenate(test_true) if test_true else np.array([])
        test_pred = np.concatenate(test_pred) if test_pred else np.array([])
        test_prob = np.concatenate(test_prob) if test_prob else np.array([])
        if test_true.size:
            test_metrics = _compute_epoch_metrics(test_true, test_pred, test_prob, num_classes)
        else:
            test_metrics = {"acc": float("nan"), "auc": float("nan"), "precision": float("nan"),
                            "recall": float("nan"), "f1": float("nan")}
        history[-1].update({f"test_{k}": v for k, v in test_metrics.items()})
        if params.verbose:
            label_dist = _format_label_distribution(test_true.astype(int), num_classes) if test_true.size else "none"
            print(f"Test label distribution: {label_dist}")
            _print_metrics("Test metrics", test_metrics)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    test_correct = 0
    test_preds = []
    test_true = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            test_correct += (preds == labels).sum().item()
            test_preds.append(preds.cpu().numpy())
            test_true.append(labels.cpu().numpy())
    test_acc = test_correct / len(test_dataset)

    return model, history, test_acc, np.concatenate(test_preds), np.concatenate(test_true)


def _train_sklearn(params, model_name):
    train_dataset, val_dataset, test_dataset, num_classes = _build_datasets(params)
    X_train, y_train = _build_flat_features(train_dataset, desc="Loading train features")
    X_val, y_val = _build_flat_features(val_dataset, desc="Loading val features")
    X_test, y_test = _build_flat_features(test_dataset, desc="Loading test features")

    if model_name == "SVM":
        net = svm.SVC(kernel="linear", gamma="scale", class_weight="balanced", random_state=seed, probability=True)
    elif model_name == "FC90":
        hidden_layers = (5, 6, 7, num_classes)
        net = neural_network.MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            max_iter=1,
            solver="sgd",
            learning_rate="adaptive",
            momentum=params.momentum,
            activation="relu",
            early_stopping=False,
            random_state=seed,
            verbose=params.verbose,
            warm_start=True,
        )
    else:
        net = SGDClassifier(penalty="elasticnet", l1_ratio=0.5, random_state=seed, loss="log_loss")

    history = []
    classes = np.unique(y_train)
    if model_name in {"FC90", "ElasticNet"}:
        epoch_iter = tqdm(range(params.n_epochs), desc=f"{model_name} epochs") if params.verbose else range(params.n_epochs)
        for epoch in epoch_iter:
            net.partial_fit(X_train, y_train, classes=classes)
            train_pred = net.predict(X_train)
            val_pred = net.predict(X_val)
            test_pred = net.predict(X_test)
            train_prob = _sklearn_predict_proba(net, X_train, num_classes)
            val_prob = _sklearn_predict_proba(net, X_val, num_classes)
            test_prob = _sklearn_predict_proba(net, X_test, num_classes)

            train_metrics = _compute_epoch_metrics(y_train, train_pred, train_prob, num_classes)
            val_metrics = _compute_epoch_metrics(y_val, val_pred, val_prob, num_classes)
            test_metrics = _compute_epoch_metrics(y_test, test_pred, test_prob, num_classes)
            history.append(
                dict(
                    epoch=epoch,
                    train=train_metrics,
                    val=val_metrics,
                    test=test_metrics,
                )
            )
            if params.verbose:
                print(f"{model_name} epoch {epoch} test label distribution: "
                      f"{_format_label_distribution(y_test.astype(int), num_classes)}")
                _print_metrics(f"{model_name} epoch {epoch} [test]", test_metrics)
    else:
        net.fit(X_train, y_train)
        train_pred = net.predict(X_train)
        val_pred = net.predict(X_val)
        test_pred = net.predict(X_test)
        train_prob = _sklearn_predict_proba(net, X_train, num_classes)
        val_prob = _sklearn_predict_proba(net, X_val, num_classes)
        test_prob = _sklearn_predict_proba(net, X_test, num_classes)
        train_metrics = _compute_epoch_metrics(y_train, train_pred, train_prob, num_classes)
        val_metrics = _compute_epoch_metrics(y_val, val_pred, val_prob, num_classes)
        test_metrics = _compute_epoch_metrics(y_test, test_pred, test_prob, num_classes)
        history.append(dict(epoch=0, train=train_metrics, val=val_metrics, test=test_metrics))
        if params.verbose:
            print(f"{model_name} test label distribution: "
                  f"{_format_label_distribution(y_test.astype(int), num_classes)}")
            _print_metrics(f"{model_name} [test]", test_metrics)

    outputs = dict(
        trainp=train_pred,
        trainy=y_train,
        valp=val_pred,
        valy=y_val,
        testp=test_pred,
        testy=y_test,
    )

    test_bacc = balanced_accuracy_score(y_test, net.predict(X_test))
    val_bacc = balanced_accuracy_score(y_val, net.predict(X_val))
    train_bacc = balanced_accuracy_score(y_train, net.predict(X_train))

    metrics = dict(train_bacc=train_bacc, val_bacc=val_bacc, test_bacc=test_bacc, history=history)
    return net, outputs, metrics


def main(args):
    params = ClassFromDict(args)
    rundate = datetime.datetime.now().strftime('%b_%d_%Y_%H_%M_%S')

    for folder in [performance_dir, models_dir, output_dir]:
        ensure_subfolder_in_folder(folder=folder, subfolder=params.model[0])

    net_preamble = '_'.join([params.model[0], rundate])

    if params.model[0] == "BNCNN":
        model, history, test_acc, test_pred, test_true = _train_bncnn(params)
        model_path = os.path.join(models_dir, params.model[0], f"{net_preamble}_net.pt")
        torch.save(model.state_dict(), model_path)

        output_path = os.path.join(output_dir, params.model[0], f"{net_preamble}_output.pkl")
        pickle.dump(
            dict(testp=test_pred, testy=test_true, history=history, test_acc=test_acc),
            open(output_path, "wb"),
        )

        perf_path = os.path.join(performance_dir, params.model[0], f"{net_preamble}_performance.json")
        with open(perf_path, "w", encoding="utf-8") as f:
            json.dump(dict(test_acc=test_acc, history=history), f, indent=2)
    else:
        model, outputs, metrics = _train_sklearn(params, params.model[0])
        model_path = os.path.join(models_dir, params.model[0], f"{net_preamble}_net.pkl")
        pickle.dump(model, open(model_path, "wb"))

        output_path = os.path.join(output_dir, params.model[0], f"{net_preamble}_output.pkl")
        pickle.dump(outputs, open(output_path, "wb"))

        perf_path = os.path.join(performance_dir, params.model[0], f"{net_preamble}_performance.json")
        with open(perf_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
