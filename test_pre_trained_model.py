import torch
import yaml
from src.train_functions import predict, plot_confusion_matrix

# NOTE: Change which model you want to test
from src.GRU_late_fusion_game import *



def main():
    # imports parameters
    with open("parameters/hyper_params.yml", "r") as f:
        config = yaml.safe_load(f)

    with open("parameters/params.yml", "r") as f:
        params = yaml.safe_load(f)
    

    # make model and get preprocessed dataset
    _, valid_loader, test_loader, model, _, _, _ = make(params, config)

    # load pre-trained weights
    # NOTE: Change which weights you want to load
    model_dict = torch.load('models/MM_GRU_game/' + config['model_GRU_game'])
    model._load_weights(model_dict)

    # Evaluation (validation)
    y_pred, y_true, acc, valance_acc, arousal_acc, f1_weighted = predict(model, valid_loader)
    print(f"validation_ACC: {acc} \nvalidation_Valence_ACC {valance_acc} \nvalidation_Aurosal_ACC \n{arousal_acc} validation_f1_weighted \n{f1_weighted}")

    # Evaluation (test)
    y_pred, y_true, acc, valance_acc, arousal_acc, f1_weighted = predict(model, test_loader)
    print(f"ACC: {acc} \nValence_ACC {valance_acc} \nAurosal_ACC \n{arousal_acc} f1_weighted \n{f1_weighted}")
    cnf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5])
    plot_confusion_matrix(cnf_matrix, 
                            classes=['Pos_Low','Neg_Low','Pos_Med','Neg_Med','Pos_Hig','Neg_Hig'],
                            title='Confusion matrix, with normalization',
                            normalize=True,
                            save=False) 

if __name__ == '__main__':
    main()