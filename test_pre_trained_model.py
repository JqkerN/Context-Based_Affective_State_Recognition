import torch
import torch.nn as nn
# Torch packages
import torch
import torch.nn as nn

# Other packages
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import yaml
import wandb
import numpy as np

# own packages
from dataset import Dataset
from train_functions import train, predict, plot_confusion_matrix
from GRU_late_fusion_game import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark=True
print('\nUsing device:', torch.cuda.get_device_name(0))
print()


def main():
    # imports parameters
    with open("parameters/hyper_params.yml", "r") as f:
        hyper = yaml.safe_load(f)

    with open("parameters/params.yml", "r") as f:
        params = yaml.safe_load(f)
    
    # initialize wandb run
    # with wandb.init(project='Affective_Recognition', entity='iliancorneliussen', tags=["GRU_LF"]):
    with wandb.init(project='results', entity='iliancorneliussen', config=hyper, tags=["GRU_LF_game", "remove"]):
        config = wandb.config
        # make model and get preprocessed dataset
        train_loader, valid_loader, test_loader, model, criterion, optimizer, EPOCHS, subject = make(params, hyper)

        # load pre-trained weights
        model_dict = torch.load('models/MM_GRU_game/' + config['model_GRU_game'])
        model._load_weights(model_dict)

        # Evaluation (validation)
        y_pred, y_true, acc, valance_acc, arousal_acc, f1_weighted, _ = predict(model, valid_loader)
        wandb.log({"validation_ACC": acc, "validation_Valence_ACC": valance_acc, "validation_Aurosal_ACC": arousal_acc, "validation_f1_weighted": f1_weighted})

        # Evaluation (test)
        y_pred, y_true, acc, valance_acc, arousal_acc, f1_weighted, y_pred_softmax = predict(model, test_loader)
        wandb.log({"ACC": acc, "Valence_ACC": valance_acc, "Aurosal_ACC": arousal_acc, "f1_weighted": f1_weighted})
        cnf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5])
        plot_confusion_matrix(cnf_matrix, 
                              classes=['Pos_Low','Neg_Low','Pos_Med','Neg_Med','Pos_Hig','Neg_Hig'],
                              title='Confusion matrix, with normalization',
                              normalize=True,
                              save=True) 

if __name__ == '__main__':
    main()