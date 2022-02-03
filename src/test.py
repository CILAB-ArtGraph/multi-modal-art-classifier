import mlflow
from functools import wraps
import argparse
from utils import track_params

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default='baseline', help='Experiment name.')
parser.add_argument('--label', type=str, default='genre', help='Label to predict (style|genre).')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=3e-4, help='Initial learning rate.')
args = parser.parse_args()

exp = 'test'
label = 'label'
epochs = 10
lr = 3e-5
verbose = True

def tracking(verbose):
    def dec_tracking(fun):
        @wraps(fun)
        def wrapper(*args, **kwargs):
            loss, acc, epoch = fun(*args, **kwargs)
            print('Running train')
            if verbose == True:
                mlflow.log_metric(f'train loss', loss, step=epoch)
                mlflow.log_metric(f'train acc', acc, step=epoch)
            return loss, acc, epoch
        return wrapper
    return dec_tracking

@tracking(verbose)
def train(epoch):
    loss = epoch
    acc = epoch

    return loss, acc, epoch

def valid():
    loss = 1
    acc = 1

    return loss, acc

def training_process():
    if verbose == True:
        track_params(args)
    for epoch in range(epochs):
        loss_train, acc_train, _ = train(epoch)

training_process()