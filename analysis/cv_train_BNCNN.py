import datetime
import os
import pickle
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data.dataset
import xarray as xr
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import mean_absolute_error as mae
from torch.autograd import Variable
from tqdm import tqdm

from utils.util_args import set_names, metrics, models_dir, performance_dir, output_dir
from utils.util_classes import ClassFromDict, PervaizBNCNN, HeSexBNCNN, He58behaviorsBNCNN, KawaharaBNCNN, HCPDataset, \
    TransformedData
from utils.util_funcs import ensure_subfolder_in_folder, get_training_params


def compute_fc_from_signal(signal_2d):
    fc = np.corrcoef(signal_2d)
    fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
    return fc.astype(np.float32)


class ADNIRandomCropDataset(torch.utils.data.Dataset):
    def __init__(self, files, labels, label_to_int, crop_length):
        self.files = files
        self.labels = labels
        self.label_to_int = label_to_int
        self.crop_length = crop_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        ts = np.load(self.files[idx]).astype(np.float32)
        n_roi, n_time = ts.shape
        if n_time < self.crop_length:
            raise ValueError(f'{os.path.basename(self.files[idx])} has {n_time} frames, crop_length={self.crop_length}')
        start = np.random.randint(0, n_time - self.crop_length + 1)
        fc = compute_fc_from_signal(ts[:, start:start + self.crop_length])
        x = torch.FloatTensor(fc[None, :, :])
        y = torch.FloatTensor([self.label_to_int[self.labels[idx]]])
        return x, y


def train_adni_bncnn(params):
    rundate = datetime.datetime.now().strftime('%b_%d_%Y_%H_%M_%S')
    torch.set_num_threads(params.n_threads)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_cuda = torch.cuda.is_available()

    net_preamble = '_'.join([params.model[0], rundate])
    for folder in [performance_dir, models_dir, output_dir]:
        ensure_subfolder_in_folder(folder=folder, subfolder=params.model[0])

    fold_keys = sorted(params.adni_fold_splits.keys(), key=int)
    for fold_key in fold_keys[params.start_fold:params.end_fold]:
        fold = int(fold_key)
        split = params.adni_fold_splits[fold_key]
        train_files = split['train']
        val_files = split['val']
        test_files = split['test']

        path_to_label = dict(zip(params.adni_files, params.adni_labels))
        train_labels = [path_to_label[x] for x in train_files]
        val_labels = [path_to_label[x] for x in val_files]
        test_labels = [path_to_label[x] for x in test_files]

        print(f'\nTraining ADNI fold {fold}')
        print(f'  train/val/test subjects: {len(train_files)}/{len(val_files)}/{len(test_files)}')

        trainset = ADNIRandomCropDataset(train_files, train_labels, params.label_to_int, params.crop_length)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, pin_memory=True)

        # instantiate from first sample
        sample_x, _ = trainset[0]
        sample_x = sample_x.unsqueeze(0)

        switcher = dict(kawahara=KawaharaBNCNN,
                        he_sex=HeSexBNCNN,
                        pervaiz=PervaizBNCNN,
                        he_58=He58behaviorsBNCNN)
        net_cls = switcher.get(params.architecture, PervaizBNCNN)

        class ADNIShape:
            multiclass = True
            n_classes = 2
            n_outcomes = 1

        net = net_cls(sample_x, ADNIShape())
        if use_cuda:
            net = net.to(device)
            cudnn.benchmark = True

        optimizer = torch.optim.SGD(net.parameters(), lr=params.lr, momentum=params.momentum, nesterov=True,
                                    weight_decay=params.wd)
        criterion = nn.BCELoss()

        def evaluate_subjects(files, labels):
            if len(files) == 0:
                return np.nan
            subject_votes = []
            true_labels = []
            net.eval()
            for fp, y_name in zip(files, labels):
                ts = np.load(fp).astype(np.float32)
                n_time = ts.shape[1]
                if n_time < params.crop_length:
                    continue
                starts = range(0, n_time - params.crop_length + 1)
                preds = []
                for start in starts:
                    fc = compute_fc_from_signal(ts[:, start:start + params.crop_length])
                    xin = torch.FloatTensor(fc[None, None, :, :])
                    if use_cuda:
                        xin = xin.to(device)
                    with torch.no_grad():
                        prob = net(xin).detach().cpu().numpy().squeeze()
                    if np.ndim(prob) == 0:
                        pred = int(prob >= 0.5)
                    else:
                        pred = int(np.argmax(prob))
                    preds.append(pred)
                vote = int(np.mean(preds) >= 0.5)
                subject_votes.append(vote)
                true_labels.append(params.label_to_int[y_name])
            acc = balanced_accuracy_score(true_labels, subject_votes)
            return acc

        best_state = None
        best_test_acc = -np.inf
        best_output = {}

        for epoch in tqdm(range(params.n_epochs), desc=f'fold {fold} epochs'):
            net.train()
            running_loss = 0.0
            for inputs, targets in trainloader:
                if use_cuda:
                    inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                if outputs.ndim > 1 and outputs.shape[-1] == 2:
                    targets = torch.nn.functional.one_hot(targets.long().squeeze(-1), num_classes=2).float()
                loss = criterion(outputs, targets.view(outputs.size()))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=params.max_norm)
                optimizer.step()
                running_loss += loss.item()

            val_acc = evaluate_subjects(val_files, val_labels)
            test_acc = evaluate_subjects(test_files, test_labels)
            print(f'Epoch {epoch:03d} | train_loss={running_loss / max(1, len(trainloader)):.4f} '
                  f'| val_bacc={val_acc:.4f} | test_bacc={test_acc:.4f}')

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_state = net.state_dict()
                best_output = dict(best_test_balanced_accuracy=test_acc, best_val_balanced_accuracy=val_acc, epoch=epoch)

        net_path = os.path.join(models_dir, params.model[0], '_'.join([net_preamble + f'fold{fold}_net.pt']))
        torch.save(best_state, net_path)
        output_path = os.path.join(output_dir, params.model[0], '_'.join([net_preamble, f'fold{fold}_output.pkl']))
        pickle.dump(best_output, open(output_path, "wb"))

    print(f'ADNI training done. Splits file: {params.split_txt_path}')
    return dict()


def main(args):
    params = ClassFromDict(args)
    if hasattr(params, 'adni_fold_splits'):
        return train_adni_bncnn(params)

    X = params.X
    Y = params.Y
    rundate = datetime.datetime.now().strftime('%b_%d_%Y_%H_%M_%S')

    # ignore possible warnings from correlation and early stopping calculation
    warnings.filterwarnings("ignore", message='An input array is constant')
    warnings.filterwarnings("ignore", message='Mean of empty slice')
    warnings.filterwarnings("ignore", message=' invalid value encountered in less_equal')

    # hardware params
    torch.set_num_threads(params.n_threads)  # limits CPU usage
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # single GPU
    use_cuda = torch.cuda.is_available()

    # identifiers for saving
    net_preamble = '_'.join([params.model[0], rundate])
    for folder in [performance_dir, models_dir, output_dir]:
        ensure_subfolder_in_folder(folder=folder, subfolder=params.model[0])

    # create structures to hold performance metrics
    folds_from_start_to_end = params.end_fold - params.start_fold
    range_folds = range(params.start_fold, params.end_fold)
    best_test_epochs = dict(zip([f'best_test_epoch_fold_{i}' for i in range_folds],
                                np.full(folds_from_start_to_end, np.nan)))
    stopped_epochs = dict(zip([f'stopped_epoch_fold_{i}' for i in range_folds],
                              np.full(folds_from_start_to_end, np.nan)))

    coords = [range(params.n_epochs), set_names, metrics, params.outcome_names, range_folds]
    coords_len = [len(x) for x in coords]
    performance = xr.DataArray(np.full(coords_len, np.nan), coords=coords,
                               dims=['epoch', 'set', 'metrics', 'outcome', 'cv_fold'])

    net_ids = []  # aim: track id of each fold's net

    for fold in range(params.start_fold, params.end_fold):

        print(f'\nTraining fold {fold}')

        transformed_data = TransformedData(fold, X, Y)
        transformed_data.preprocess_data()

        trainset = HCPDataset(transformed_data, mode="train")
        testset = HCPDataset(transformed_data, mode="test")
        valset = HCPDataset(transformed_data, mode="val")

        dataloader_kwargs = dict(shuffle=True, batch_size=8, pin_memory=True)
        trainloader = torch.utils.data.DataLoader(trainset, **dataloader_kwargs)
        testloader = torch.utils.data.DataLoader(testset, **dataloader_kwargs)
        valloader = torch.utils.data.DataLoader(valset, **dataloader_kwargs)

        print("\nInit Network",
              f'\nTraining data: {", ".join(transformed_data.matrix_labels)}',
              f'\nPredicting: {", ".join(transformed_data.outcome_names)}')

        def instantiate_net(architecture, train_set):
            switcher = dict(kawahara=KawaharaBNCNN,
                            he_sex=HeSexBNCNN,
                            pervaiz=PervaizBNCNN,
                            he_58=He58behaviorsBNCNN)
            net = switcher.get(architecture, PervaizBNCNN)
            return net(train_set, transformed_data)

        net = instantiate_net(params.architecture, trainset.X)

        if use_cuda:
            net = net.to(device)
            assert next(net.parameters()).is_cuda, 'Parameters are not on the GPU !'
            cudnn.benchmark = True

        assert id(net) not in net_ids, 'No new net was instantiated. Please debug.'
        net_ids.append(id(net))

        def init_weights_he(m):
            """ Weights initialization for the dense layers using He Uniform initialization.
            Only applies to linear layers
             (He et al., 2015) http://arxiv.org/abs/1502.01852, https://keras.io/initializers/#he_uniform
        """
            if type(m) == torch.nn.Linear:
                fan_in = net.dense1.in_features
                he_lim = np.sqrt(6 / fan_in)
                m.weight.data.uniform_(-he_lim, he_lim)

        net.apply(init_weights_he)

        def get_optimizer():
            if params.optimizer == 'sgd':
                optimizer = torch.optim.SGD(net.parameters(), lr=params.lr, momentum=params.momentum, nesterov=True,
                                            weight_decay=params.wd)
            elif params.optimizer == 'adam':
                optimizer = torch.optim.Adam(net.parameters(), lr=params.lr, weight_decay=params.wd)

            return optimizer

        optimizer = get_optimizer()

        def get_criterion():
            if Y.multiclass:
                if Y.n_classes == 2:
                    criterion = nn.BCELoss(
                        weight=torch.Tensor(Y.class_weights).to(device))  # balanced Binary Cross Entropy
                elif Y.n_classes > 2:
                    criterion = nn.CrossEntropyLoss(weight=torch.Tensor(Y.class_weights).to(device))
            else:
                criterion = torch.nn.MSELoss()

            return criterion

        criterion = get_criterion()

        def train_net():  # training in mini batches
            net.train()
            running_loss = 0.0

            y_pred = []
            y_true = []

            for batch_idx, (inputs, targets) in enumerate(trainloader):

                if use_cuda:
                    if not Y.multioutcome and not Y.multiclass:
                        inputs, targets = inputs.to(device), targets.to(device).unsqueeze(
                            1)  # unsqueeze needed for vstack
                    else:
                        inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                inputs, targets = Variable(inputs), Variable(targets)

                outputs = net(inputs)
                targets = targets.view(outputs.size())

                if Y.multiclass and Y.n_classes > 2:
                    loss = criterion(input=outputs, target=torch.argmax(targets.data, 1))
                else:
                    loss = criterion(input=outputs, target=targets)

                loss.backward()

                # prevents a vanishing / exploding gradient problem
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=params.max_norm)

                for p in net.parameters():
                    p.data.add_(-params.lr, p.grad.data)

                optimizer.step()
                running_loss += loss.data.mean(0)

                y_pred.append(outputs.data.cpu().numpy())
                y_true.append(targets.data.cpu().numpy())

                if batch_idx == len(trainloader) - 1:  # print loss for final batch
                    if params.verbose:
                        print(f'\n train loss: %.6f' % (running_loss / len(trainloader)))

            if not Y.multioutcome and not Y.multiclass:
                return np.vstack(y_pred), np.vstack(y_true), running_loss / batch_idx
            else:
                return np.vstack(y_pred), np.vstack(y_true).squeeze(), running_loss / batch_idx

        def evaluate_net(set_name):
            if set_name == 'test':
                dataloader = testloader
            elif set_name == 'val':
                dataloader = valloader

            net.eval()
            running_loss = 0.0

            y_pred = []
            y_true = []

            for batch_idx, (inputs, targets) in enumerate(dataloader):
                if use_cuda:
                    if not Y.multioutcome and not Y.multiclass:
                        inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)
                    else:
                        inputs, targets = inputs.to(device), targets.to(device)

                with torch.no_grad():
                    inputs, targets = Variable(inputs), Variable(targets)

                    outputs = net(inputs)
                    targets = targets.view(outputs.size())

                    if Y.multiclass and Y.n_classes > 2:
                        loss = criterion(input=outputs, target=torch.argmax(targets.data, 1))
                    else:
                        loss = criterion(input=outputs, target=targets)

                    y_pred.append(outputs.data.cpu().numpy())
                    y_true.append(targets.data.cpu().numpy())

                running_loss += loss.data.mean(0)

                if batch_idx == len(dataloader) - 1:  # print loss for final batch
                    if params.verbose:
                        print(f'\n {set_name} loss: %.6f' % (running_loss / len(dataloader)))

            if not Y.multioutcome and not Y.multiclass:
                return np.vstack(y_pred), np.vstack(y_true), running_loss / batch_idx
            else:
                return np.vstack(y_pred), np.vstack(y_true).squeeze(), running_loss / batch_idx

        output_names = [[f'{name}p', f'{name}y'] for name in set_names]
        output_names = [x for y in output_names for x in y]
        epoch_output = dict(zip(output_names, [[] for _ in range(len(output_names))]))

        for epoch in range(params.n_epochs):

            print("\nEpoch %d" % epoch)

            test_mae_kwargs = dict(set='test', metrics='MAE', cv_fold=fold)
            test_r_kwargs = dict(set='test', metrics='pearsonr', cv_fold=fold)
            test_acc_kwargs = dict(set='test', metrics='accuracy', cv_fold=fold)

            def calculate_and_print_performance(pred, true, set_name):

                if Y.multiclass:
                    acc_kwargs = dict(epoch=epoch, metrics=['accuracy'], cv_fold=fold)
                    pred, true = np.argmax(pred, 1), np.argmax(true, 1)
                    acc = balanced_accuracy_score(true, pred)
                    performance.loc[{'set': set_name, **acc_kwargs}] = acc

                    print(f"  {Y.outcome_names}, {set_name} accuracy : {acc:.3}")

                else:
                    save_metrics = ['MAE', 'pearsonr', 'pearsonp', 'spearmanr', 'spearmanp']
                    metrics_kwargs = dict(epoch=epoch, metrics=save_metrics, cv_fold=fold)

                    if Y.multioutcome:
                        mae_all = np.array([mae(true[:, i], pred[:, i]) for i in range(Y.n_outcomes)])
                        pears_all = np.array([list(pearsonr(true[:, i], pred[:, i])) for i in range(Y.n_outcomes)])
                        spear_all = np.array([list(spearmanr(true[:, i], pred[:, i])) for i in range(Y.n_outcomes)])
                        performance.loc[{'set': set_name, **metrics_kwargs}] = [mae_all,
                                                                                pears_all[:, 0], pears_all[:, 1],
                                                                                spear_all[:, 0], spear_all[:, 1]]

                        for i in range(Y.n_outcomes):
                            print(f"  {Y.outcome_names[i]}, {set_name} MAE : {mae_all[i]:.3},"
                                  f" pearson R: {pears_all[i, 0]:.3} (p = {pears_all[i, 1]:.4})")

                    else:
                        mae_one = mae(pred, true)
                        pears_one = pearsonr(pred[:, 0], true[:, 0])
                        spear_one = spearmanr(pred[:, 0], true[:, 0])
                        performance.loc[{'set': set_name, **metrics_kwargs}] = np.array([mae_one, pears_one[0],
                                                                                         pears_one[1], spear_one[0],
                                                                                         spear_one[1]])[:, None]

                        print(f"  {Y.outcome_names[0]}, {set_name} MAE : {mae_one:.3}, "
                              f"pearson R: {pears_one[0]:.3} (p = {pears_one[1]:.4})")

            # get model output
            for name in set_names:
                if name == 'train':
                    pred, true, loss = train_net()
                else:
                    pred, true, loss = evaluate_net(name)

                performance.loc[dict(epoch=epoch, set=name, metrics='loss', cv_fold=fold)] = [loss]
                epoch_output[f'{name}p'] = pred
                epoch_output[f'{name}y'] = true
                calculate_and_print_performance(pred, true, name)

            def best_test_epoch():
                if Y.multiclass:
                    best_epoch_yet = performance.loc[test_acc_kwargs].argmax().values
                elif Y.multioutcome:
                    best_epoch_yet = performance.loc[test_mae_kwargs].mean(axis=-1).argmin().values
                else:
                    best_epoch_yet = performance.loc[test_mae_kwargs].argmin().values

                return best_epoch_yet

            if best_test_epoch() == epoch:
                best_epoch = epoch
                best_net = net.state_dict()
                best_output = epoch_output.copy()

            def is_performance_stagnating():
                if Y.multiclass:
                    recent_acc = performance[epoch - params.ep_int:-1].loc[test_acc_kwargs]
                    current_acc = performance[epoch].loc[test_acc_kwargs]
                    stagnant = np.nanmean(recent_acc >= current_acc)

                else:
                    recent_mae = performance[epoch - params.ep_int:-1].loc[test_mae_kwargs]
                    current_mae = performance[epoch].loc[test_mae_kwargs]

                    recent_r = performance[epoch - params.ep_int:-1].loc[test_r_kwargs]
                    current_r = performance[epoch].loc[test_r_kwargs]

                    if Y.multioutcome:  # stagnant if model stops learning on at least half of outcomes
                        majority = int(np.ceil(Y.n_outcomes / 2))
                        stagnant_mae = (np.nanmean(recent_mae, axis=0) <= current_mae).sum() >= majority
                        stagnant_abs_r = (np.nanmean(np.abs(recent_r), axis=0) <= np.abs(current_r)).sum() >= majority

                    else:
                        stagnant_mae = np.nanmean(recent_mae, axis=0) <= current_mae
                        stagnant_abs_r = np.nanmean(np.abs(recent_r), axis=0) <= np.abs(current_r)

                    stagnant = bool(stagnant_mae + stagnant_abs_r)

                return stagnant

            if params.early and epoch > params.min_train_epochs:
                if is_performance_stagnating():
                    if params.verbose:
                        print('\bEarly stopping conditions reached, stopping training...')
                    stopped_epochs[f'stopped_epoch_fold_{fold}'] = epoch - params.ep_int
                    break

        best_test_epochs[f'best_test_epoch_fold_{fold}'] = best_epoch

        # saving net weights and output
        net_path = os.path.join(models_dir, params.model[0], '_'.join([net_preamble + f'fold{fold}_net.pt']))
        torch.save(best_net, net_path)
        output_path = os.path.join(output_dir, params.model[0], '_'.join([net_preamble, f'fold{fold}_output.pkl']))
        pickle.dump(best_output, open(output_path, "wb"))

    training_params = get_training_params(params=params, transformed_data=transformed_data)
    training_params.update({'rundate': rundate})
    training_params.update(best_test_epochs)
    if params.early:
        training_params.update(stopped_epochs)

    filename_performance = '_'.join([net_preamble, 'performance.nc'])

    performance.attrs = training_params
    performance.name = filename_performance
    performance.to_netcdf(os.path.join(performance_dir, params.model[0], filename_performance))  # saving

    def calculate_and_print_val_performance(best_test_epochs):

        print(f'\nModel training i/o:'
              f'\nmatrix label(s): {", ".join(transformed_data.matrix_labels)}'
              f'\noutcome(s): {", ".join(transformed_data.outcome_names)}'
              f'\ntransformation: {transformed_data.transformations}'
              f'\nconfound(s): {", ".join(list(filter(None, transformed_data.confound_names)))}'
              f'\nbest test epoch(s): {list(best_test_epochs.values())}')

        print(f'\nValidation set performance on best test epoch (mean across cv_folds):')
        for metric in metrics:
            best_val_metric_by_fold = [performance.loc[dict(set='val',
                                                            metrics=metric,
                                                            epoch=best_test_epochs[f'best_test_epoch_fold_{i}'],
                                                            cv_fold=i)].values.tolist() for i in range_folds]
            best_val_metric_mean = np.nanmean(best_val_metric_by_fold, axis=0)
            print(f"  {metric}: {best_val_metric_mean}")

    calculate_and_print_val_performance(best_test_epochs)

    return dict(performance=performance)


if __name__ == '__main__':
    main()
