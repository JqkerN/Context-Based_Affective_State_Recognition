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

# own packages
from src.dataset import Dataset
from src.train_functions import train, predict, plot_confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark=True
print('\nUsing device:', torch.cuda.get_device_name(0))
print()


###################################################################
##################         GRU-RNN          #######################
###################################################################

class GRU(nn.Module):
    def __init__(self, input_size_face, input_size_pose, input_size_smile, input_size_game, hidden_size, num_layers, num_classes, dropout=0.2):
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.best_loss = 10000
        
        # -> input_size needs to be: (batch_size, seq, input_size)
        self.gru_face = nn.GRU(input_size_face, hidden_size, num_layers, batch_first=True)
        self.gru_pose = nn.GRU(input_size_pose, hidden_size, num_layers, batch_first=True)
        self.gru_smile = nn.GRU(input_size_smile, hidden_size, num_layers, batch_first=True)
        self.gru_game = nn.GRU(input_size_game, hidden_size, num_layers, batch_first=True)

        self.activation = nn.LeakyReLU(0.2)
        self.SF = nn.Linear(hidden_size*4, hidden_size)
        self.prediction = nn.Linear(hidden_size, num_classes)

        
    def forward(self, face, pose, smile, game):
        # FACE
        h0_face = torch.zeros(self.num_layers, face.size(0), self.hidden_size).to(device) 
        x_face, _ = self.gru_face(face, h0_face)  
        x_face = x_face[:, -1, :]  
        x_face = self.activation(x_face)

        # POSE
        h0_pose = torch.zeros(self.num_layers, pose.size(0), self.hidden_size).to(device) 
        x_pose, _ = self.gru_pose(pose, h0_pose)
        x_pose = x_pose[:, -1, :]  
        x_pose = self.activation(x_pose)

        # SMILE
        h0_smile = torch.zeros(self.num_layers, smile.size(0), self.hidden_size).to(device) 
        x_smile, _ = self.gru_smile(smile, h0_smile)  
        x_smile = x_smile[:, -1, :]
        x_smile = self.activation(x_smile)

        # GAME
        h0_game = torch.zeros(self.num_layers, game.size(0), self.hidden_size).to(device) 
        x_game, _ = self.gru_game(game, h0_game)  
        x_game = x_game[:, -1, :]
        x_game = self.activation(x_game)

        # Late Fusion
        x = torch.cat((x_face, x_pose, x_smile, x_game), 1)
        x = self.SF(x)

        # Prediction layer
        x = self.activation(x)
        x = self.prediction(x)
        return x


    def _load_weights(self, pre_trained_dict):
        model_dict = self.state_dict()
        # 1. filter out unnecessary keys
        pre_trained_dict = {k: v for k, v in pre_trained_dict['model_state_dict'].items() if k in model_dict}
        print(f"\n--- Adding pre-trained layer:\n\t{pre_trained_dict.keys()}")
        # 2. overwrite entries in the existing state dict
        model_dict.update(pre_trained_dict) 
        # 3. load the new state dict
        self.load_state_dict(model_dict)


    def _save_model(self, loss, f1, epoch, optimizer):
        if loss < self.best_loss:
            self.best_loss = loss
            logs = {# best_score-metrics
                    "epoch": epoch,
                    "best_loss": loss, 
                    "best_valid_f1_weighted": f1,
                    }
            wandb.log(logs)
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        'valid_f1_weighted': f1
                        }, 'models/MM_GRU_game/' + wandb.run.name + '.pth')



###################################################################
##################         CREATE           #######################
###################################################################

def make(params, hyper):
    """
    This function makes the dataset and nn.model
    """
    # Hyper parameters
    EPOCHS = int(hyper['num_epochs'])
    BATCH_SIZE = int(hyper['batch_size'])
    LEARNING_RATE = float(hyper['learning_rate'])
    HIDDEN_SIZE = int(hyper['hidden_size'])
    NUM_LAYERS = int(hyper['num_layers'])
    WEIGHT_DECAY = float(hyper['weight_decay'])
    NORMALIZE = hyper['normalize']
    AMSGRAD = hyper['amsgrad']
    params['Dataset']['Settings']['opensmile_window'] = hyper['opensmile_window']

    print('\n--- Hyper parameters:')
    for key in hyper.keys():
        print(f'\t{key}: {hyper[key]}')
    print()

    # Load and preprocess dataset
    SPLIT = (70,20,10) # dataset split
    data = Dataset(parameters=params['Dataset'])
    data.load_data()
    train_loader, valid_loader, test_loader, class_weights = data.preprocess_recurrent(batch_size=BATCH_SIZE,
                                                                                        normalize=NORMALIZE,
                                                                                        split=SPLIT,
                                                                                        shuffle=True,
                                                                                        stratified=True,
                                                                                        remove=True)
    
    # Static parameters
    INPUT_SIZE_FACE = len(params['Dataset']['Labels']['openface'])
    INPUT_SIZE_POSE = len(params['Dataset']['Labels']['openpose'])
    INPUT_SIZE_SMILE = len(params['Dataset']['Labels']['opensmile'])
    INPUT_SIZE_GAME = len(params['Dataset']['Labels']['game'])
    NUM_CLASSES = 6

    # Create model
    model = GRU(INPUT_SIZE_FACE, INPUT_SIZE_POSE, INPUT_SIZE_SMILE, INPUT_SIZE_GAME, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)
    model.to(device) # send to GPU

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device)) # Cross-entroy-lossfunction for multiclass classification weight=class_weights.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, amsgrad=AMSGRAD, weight_decay=WEIGHT_DECAY) # Adam Optimizer
    return train_loader, valid_loader, test_loader, model, criterion, optimizer, EPOCHS


###################################################################
##################          MAIN            #######################
###################################################################
def main(project, entity):
    # imports parameters
    with open("parameters/hyper_params.yml", "r") as f:
        hyper = yaml.safe_load(f)

    with open("parameters/params.yml", "r") as f:
        params = yaml.safe_load(f)
    
    # initialize wandb run
    with wandb.init(project=project, entity=entity, config=hyper, tags=["GRU_LF_game"]):
        config = wandb.config
        # make model and get preprocessed dataset
        train_loader, valid_loader, test_loader, model, criterion, optimizer, EPOCHS = make(params, hyper)

        # load pre-trained weights
        face_dict = torch.load('models/face/' + config['model_face'])
        model._load_weights(face_dict)
        pose_dict = torch.load('models/pose/' + config['model_pose'])
        model._load_weights(pose_dict)
        smile_dict = torch.load('models/smile/' + config['model_smile'])
        model._load_weights(smile_dict)
        game_dict = torch.load('models/game/' + config['model_game'])
        model._load_weights(game_dict)

        # train model
        wandb.watch(model, criterion, log='all', log_freq=1)
        train(model, train_loader, valid_loader, criterion, optimizer, EPOCHS)

        # Evaluation (validation)
        y_pred, y_true, acc, valance_acc, arousal_acc, f1_weighted = predict(model, valid_loader)

        wandb.log({"ACC (validation)": acc, "Valence_ACC  (validation)": valance_acc, "Aurosal_ACC  (validation)": arousal_acc, "f1_weighted  (validation)": f1_weighted})
        cnf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5])
        plot_confusion_matrix(cnf_matrix, 
                              classes=['Pos_Low','Neg_Low','Pos_Med','Neg_Med','Pos_Hig','Neg_Hig'],
                              title='Confusion matrix, with normalization',
                              normalize=True,
                              save=True) 

        artifact = wandb.Artifact('model', type='model')
        artifact.add_file('models/MM_GRU_game/' + wandb.run.name + '.pth')
        wandb.log_artifact(artifact)



    
if __name__ == '__main__':
    main()