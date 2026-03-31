import argparse
import multiprocessing
import os

import pandas as pd

from utils.util_args import multiclass_variables, input_dir, sub_info_dir

if not os.listdir(input_dir) and all(elem == '__init__.py' for elem in os.listdir(input_dir)):
    raise FileNotFoundError(f'Please add input data directory to the {input_dir} directory')

parser = argparse.ArgumentParser(description="train multiple personality-predicting models on HCP data")

parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")

# degrees of freedom in the model input/output
in_out = parser.add_argument_group('in_out', 'model I/O params')
in_out.add_argument("-on", "--outcome_names", required=False, type=str, nargs='+', default=None,
                    help="the outcome to predict")
in_out.add_argument("-md", "--matrix_directory", required=False, nargs='?',
                    help='matrix directory containing matrix input data')
in_out.add_argument("-mo", "--model", required=True, choices=['BNCNN', 'SVM', 'FC90', 'ElasticNet'],
                    type=str, help='the model to use', nargs=1)
in_out.add_argument('--architecture', required=False, choices=['pervaiz', 'he_sex', 'kawahara', 'he_58'],
                    default='pervaiz', help='BrainNetCNN architecture', nargs='?')
in_out.add_argument("-t", "--tasks", required=False, type=str, default=[None],
                    help="name of task subdirectories in matrix_directory", nargs='+')

in_out.add_argument('--input_type', choices=['matrix', 'adni_signal'], default='matrix', nargs='?',
                    help='training input type: matrix text files or ADNI N*T signal npy files')
in_out.add_argument('--adni_signal_dir', required=False, nargs='?',
                    help='directory containing ADNI .npy files shaped N*T')
in_out.add_argument('--adni_classes', required=False, nargs=2,
                    help='two class labels (filename prefixes before first underscore) for binary classification')
in_out.add_argument('--adni_label_csv', required=False, nargs='?',
                    help='csv with columns [subject, label] used to map ADNI files to labels')

# data transformation args
transforms = parser.add_argument_group('transforms', 'data transformation params')
transforms.add_argument('--transformations', choices=['untransformed', 'tangent'],
                        default='untransformed', help='data transformations to apply', nargs='?')
transforms.add_argument('--deconfound_flavor', choices=['X1Y0', 'X1Y1', 'X0Y0'], default='X0Y0',
                        help='deconfounding method', nargs='?')
transforms.add_argument('--tan_mean', choices=['euclidean', 'harmonic'], default='euclidean', nargs='?')
transforms.add_argument('-cn', '--confound_names', default=[None], type=str, nargs='+',
                        help='confounds to regress out of outcome')
transforms.add_argument("-sc", "--scale_confounds", action='store_const', default=None, const='minmax',
                        help='(boolean) confound minmax normalization to [0,1] range, applied before deconfounding')
transforms.add_argument("-sf", "--scale_features", action='store_const', default=None, const='minmax',
                        help='(boolean) feature minmax normalization to [0,1] range, applied before training')
transforms.add_argument('--use_sig_features', action='store_const', default=False, const=True,
                        help='whether to use only significant input features during oneD model training')

# training hyperparameters
hyper = parser.add_argument_group('hyper', 'model training hyperparameters')
hyper.add_argument('--optimizer', choices=['sgd', 'adam'], default='sgd', help='optimization strategy', nargs='?')
hyper.add_argument('--momentum', default=0.9, type=float, help='momentum', nargs='?')
hyper.add_argument('--lr', default=0.00001, type=float, help='learning rate',
                   nargs='?')
hyper.add_argument('--wd', default=.0005, type=float, help=' weight decay', nargs='?')
hyper.add_argument('--max_norm', default=1.5, type=float,
                   help='maximum value of normed gradients, to prevent explosion/vanishing', nargs='?')
hyper.add_argument('--families_together', action='store_const', default=False, const=True,
                   help='causes all members of a subjects\' family to be assigned to the same partition during '
                        'cross-validation')

# params for epochs to train over
epochs = parser.add_argument_group('epochs', 'training iterations params')
epochs.add_argument('--n_folds', default=5, type=int, help='# of cross validation folds', nargs='?')
epochs.add_argument('--start_fold', default=0, type=int, help='fold training starts on', nargs='?')
epochs.add_argument('--end_fold', default=5, type=int, help='fold after which training ends', nargs='?')
epochs.add_argument('--n_epochs', default=300, type=int, help='max # of epochs to train BNCNN over', nargs='?')
epochs.add_argument('-ea', '--early', action='store_true', help='early stopping boolean')
epochs.add_argument('--ep_int', type=int, default=10, help='with early stopping, if model has not improved over mean '
                                                           'performance of previous {ep_int} epochs, stop early',
                    nargs='?')
epochs.add_argument('--min_train_epochs', default=50, type=int, help='mininmum epochs to train before early stopping',
                    nargs='?')
epochs.add_argument('--crop_length', default=100, type=int, nargs='?',
                    help='sequence length (frames) to crop from ADNI signal per sample')
epochs.add_argument('--split_txt_path', default='split.txt', nargs='?',
                    help='path to ADNI split txt file; generated if missing')
epochs.add_argument('--test_size', default=0.2, type=float, nargs='?',
                    help='test split proportion for ADNI single-run training')

# hardware parameters
hardware = parser.add_argument_group('hardware', 'hardware params')
hardware.add_argument('--n_threads', default=int(multiprocessing.cpu_count() / 2),
                      help='default number of threads to use during training', nargs='?')

uncond_args = parser.parse_args()  # unconditional args


def set_conditional_args():
    multi_input = len(uncond_args.tasks) != 1

    if uncond_args.input_type == 'adni_signal':
        matrix_labels = ['adni_dynamic_fc']
        uncond_args.tasks = ['adni_signal']
    elif any(uncond_args.tasks):
        matrix_labels = ['_'.join([uncond_args.matrix_directory, task]) for task in uncond_args.tasks if task]
    else:
        matrix_labels = uncond_args.matrix_directory
        uncond_args.tasks = [uncond_args.matrix_directory]
    uncond_args.end_fold = uncond_args.n_folds

    conditional = parser.add_argument_group('conditionals', 'conditional arguments')
    conditional.add_argument('n_input', action='store_const', const=len(uncond_args.tasks),
                             help='number of input matrix datasets to train on')
    conditional.add_argument('multi_input', action='store_const', const=multi_input,
                             help='bool for multi input training')
    conditional.add_argument('matrix_labels', action='store_const', const=matrix_labels,
                             help='names of input matrix datasets')


def exit_logic():
    multioutcome = len(uncond_args.outcome_names) > 1 if uncond_args.outcome_names else False

    if uncond_args.input_type == 'adni_signal':
        if not uncond_args.adni_signal_dir or not os.path.isdir(uncond_args.adni_signal_dir):
            raise NotADirectoryError('For adni_signal input, --adni_signal_dir must be a valid directory')
        if not uncond_args.adni_label_csv or not os.path.isfile(uncond_args.adni_label_csv):
            raise FileNotFoundError('For adni_signal input, --adni_label_csv must be a valid csv file')
        if uncond_args.adni_classes and len(uncond_args.adni_classes) != 2:
            raise ValueError('If provided, --adni_classes must include exactly two labels')
        if uncond_args.model != ['BNCNN']:
            raise ValueError('ADNI signal mode currently supports BNCNN training only')
        if not 0 < uncond_args.test_size < 1:
            raise ValueError('--test_size must be in (0,1)')
        uncond_args.n_folds = 1
        uncond_args.start_fold = 0
        uncond_args.end_fold = 1
        return

    if not uncond_args.matrix_directory:
        raise ValueError('matrix_directory is required when input_type is matrix')
    if not uncond_args.outcome_names:
        raise ValueError('outcome_names (-on) is required when input_type is matrix')

    mat_dir = os.path.join(input_dir, uncond_args.matrix_directory)
    task_dirs = [item for item in os.listdir(mat_dir) if os.path.isdir(os.path.join(mat_dir, item))]
    subject_info = pd.read_csv(f'{input_dir}/{sub_info_dir}/{uncond_args.matrix_directory}_subject_info.csv')
    info_columns = subject_info.columns.to_list()

    for task in uncond_args.tasks:
        task = task if task else mat_dir
        if task not in task_dirs:
            raise NotADirectoryError(f'\'{task}\' subdirectory is not available in {mat_dir}')

    for task_dir in task_dirs:
        task_dir = task_dir if task_dir else mat_dir
        if not os.listdir(os.path.join(mat_dir, task_dir)):
            raise FileNotFoundError(f'Please add data to \'/{mat_dir}/{task_dir}/\'')

    if not all([outcome in info_columns for outcome in uncond_args.outcome_names]):
        raise ValueError(f"Not all outcomes are available in {uncond_args.matrix_directory}_subject_info.csv")

    if any(uncond_args.confound_names):
        if not all([confound in info_columns for confound in uncond_args.confound_names]):
            raise ValueError(f"Not all confounds are available in {uncond_args.matrix_directory}_subject_info.csv")

    if uncond_args.deconfound_flavor != 'X0Y0' and not any(uncond_args.confound_names):
        raise ValueError('If deconfounding is desired, please choose at least one confound')

    if uncond_args.n_folds < 3:
        raise ValueError(0, 'Cross validation folds must be >= 3')

    if 'SVM' in uncond_args.model and multioutcome:
        raise ValueError('SVM cannot handle multi-outcome problems')

    if any([x in multiclass_variables for x in uncond_args.outcome_names]) and multioutcome:
        raise ValueError('Only single-outcome prediction is supported for multiclass outcomes')

    if uncond_args.architecture == 'he_sex' and uncond_args.outcome_names != ['Gender']:
        raise ValueError('\'he_sex\' architecture only accommodates outcome \'Gender\'')

    if uncond_args.start_fold < 0:
        raise ValueError('start_fold must be non-negative')

    if uncond_args.end_fold <= uncond_args.start_fold:
        raise ValueError('end_fold must be > start_fold')

    if uncond_args.start_fold >= uncond_args.n_folds:
        raise ValueError('start_fold must be < n_folds')

    if uncond_args.end_fold > uncond_args.n_folds:
        raise ValueError('end_fold must be <= n_folds')


set_conditional_args()
exit_logic()

args = parser.parse_args()
pargs = vars(args)  # dict of passed args

# printing i/o of model
if args.verbose:
    print('\nParser arguments:', pargs, '\n')


def train_models():
    if args.verbose:
        print("\nTraining %s to predict %s from %s directory, with task(s) %s...\n" %
              (args.model, args.outcome_names, args.matrix_directory, args.tasks))

    if args.input_type == 'adni_signal':
        from preprocessing import load_adni_data
        pargs.update(load_adni_data.main(pargs))
    else:
        from preprocessing import load_data
        pargs.update(load_data.main(pargs))

    if args.model == ['BNCNN']:
        from analysis import cv_train_BNCNN
        cv_train_BNCNN.main(pargs)

    elif any(model in args.model for model in ['SVM', 'FC90', 'ElasticNet']):
        from analysis import cv_train_1D_networks
        cv_train_1D_networks.main(pargs)

    print(f'\n{args.model[0]} training done!\n')


train_models()
