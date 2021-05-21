import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm
from sklearn.metrics import f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def multi_acc(y_pred, y):
    """
    This function gives multiclass accuracys
    """
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    # Accuracy
    correct_pred = (y_pred_tags == y).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc * 100).item()

    # valence & arousal
    valence_true = 0
    valence_pred = 0
    arousal_true = 0
    arousal_pred = 0
    for i, pred_tag in enumerate(y_pred_tags):
        tag = y[i]
        # Valence
        if tag == 0 or tag == 2 or tag == 4:
            valence_true += 1
            if pred_tag == 0 or pred_tag == 2 or pred_tag == 4:
                valence_pred += 1
        elif tag == 1 or tag == 3 or tag == 5:
            valence_true += 1
            if pred_tag == 1 or pred_tag == 3 or pred_tag == 5:
                valence_pred += 1

        # Arousal
        if tag == 0 or tag == 1:
            arousal_true += 1
            if pred_tag == 0 or pred_tag == 1:
                arousal_pred += 1
        elif tag == 2 or tag == 3:
            arousal_true += 1
            if pred_tag == 2 or pred_tag == 3:
                arousal_pred += 1
        elif tag == 4 or tag == 5:
            arousal_true += 1
            if pred_tag == 4 or pred_tag == 5:
                arousal_pred += 1

    valance_acc = round(valence_pred/valence_true * 100)
    arousal_acc = round(arousal_pred/arousal_true * 100)
    np_y = y.detach().cpu().numpy()
    np_y_pred_tags = y_pred_tags.detach().cpu().numpy()
    f1_weighted = f1_score(np_y, np_y_pred_tags, average="weighted")
    return acc, valance_acc, arousal_acc, f1_weighted


def train(model, train_loader, valid_loader, criterion, optimizer, num_epochs):
    """
    This function trains and validate the model
    """
    print(model)
    for epoch in tqdm(range(1, num_epochs+1), desc='Training progress'):
        # TRAINING
        model.train()
        train_epoch_loss = 0
        train_epoch_acc = 0
        train_epoch_acc_valance = 0
        train_epoch_acc_arousal = 0
        train_epoch_f1 = 0
        optimizer.zero_grad()
        for train_face, train_pose, train_smile, train_game, y_train in train_loader: 
            # Load to device 
            train_face = train_face.to(device)
            train_pose = train_pose.to(device)
            train_smile = train_smile.to(device)
            train_game = train_game.to(device)
            y_train = y_train.to(device)

            # Forward
            y_train_pred = model(train_face, train_pose, train_smile, train_game)
            train_loss = criterion(y_train_pred, y_train)
            train_acc, train_acc_valance, train_acc_arousal, train_f1 = multi_acc(y_train_pred, y_train)
            
            # Backward
            train_loss.backward()

            # Extract metrics
            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc
            train_epoch_acc_valance += train_acc_valance
            train_epoch_acc_arousal += train_acc_arousal
            train_epoch_f1 += train_f1

        optimizer.step()

        # VALIDATION
        with torch.no_grad():
            model.eval()
            valid_epoch_loss = 0
            valid_epoch_acc = 0
            valid_epoch_acc_valance = 0
            valid_epoch_acc_arousal = 0
            valid_epoch_f1 = 0
            for valid_face, valid_pose, valid_smile, valid_game, y_valid in valid_loader:
                # Load to device 
                valid_face = valid_face.to(device)
                valid_pose = valid_pose.to(device)
                valid_smile = valid_smile.to(device)
                valid_game = valid_game.to(device)
                y_valid = y_valid.to(device)

                # Forward
                y_valid_pred = model(valid_face, valid_pose, valid_smile, valid_game)
                valid_loss = criterion(y_valid_pred, y_valid)
                valid_acc, valid_acc_valance, valid_acc_arousal, valid_f1 = multi_acc(y_valid_pred, y_valid)

                # Extract metrics
                valid_epoch_loss += valid_loss.item()
                valid_epoch_acc += valid_acc
                valid_epoch_acc_valance += valid_acc_valance
                valid_epoch_acc_arousal += valid_acc_arousal
                valid_epoch_f1 += valid_f1
        model._save_model(loss=valid_epoch_loss/len(valid_loader), 
                          f1=valid_epoch_f1/len(valid_loader), 
                          epoch=epoch, 
                          optimizer=optimizer)
        logs = {# loss-metrics
                "epoch": epoch,
                "train_loss": train_epoch_loss/len(train_loader), 
                "valid_loss": valid_epoch_loss/len(valid_loader),
                # train-metrics
                "Train_ACC": train_epoch_acc/len(train_loader), 
                "Train_Valence_ACC": train_epoch_acc_valance/len(train_loader), 
                "Train_Arousal_ACC": train_epoch_acc_arousal/len(train_loader),
                "Train_f1_weighted": train_epoch_f1/len(train_loader),
                # valid-metrics
                "Valid_ACC": valid_epoch_acc/len(valid_loader), 
                "Valid_Valence_ACC": valid_epoch_acc_valance/len(valid_loader), 
                "Valid_Arousal_ACC": valid_epoch_acc_arousal/len(valid_loader),
                "Valid_f1_weighted": valid_epoch_f1/len(valid_loader)}
        wandb.log(logs) # Send logs to Weights and Biases


def predict(model, loader):
    """
    This function outputs prediciton values and accuracy metrics.
    """
    with torch.no_grad():
        model.eval()
        y_pred_list = list()
        y_true_list = list()
        pred_epoch_acc = 0
        pred_epoch_acc_valance = 0
        pred_epoch_acc_arousal = 0
        pred_epoch_pred_f1 = 0
        for pred_face, pred_pose, pred_smile, pred_game, y_true in loader:
            # Load to device 
            pred_face = pred_face.to(device)
            pred_pose = pred_pose.to(device)
            pred_smile = pred_smile.to(device)
            pred_game = pred_game.to(device)
            y_true = y_true.long().to(device)

            # Forward
            y_pred = model(pred_face, pred_pose, pred_smile, pred_game)
            pred_acc, pred_acc_valance, pred_acc_arousal, pred_f1 = multi_acc(y_pred, y_true)

            # Extract metrics
            pred_epoch_acc += pred_acc
            pred_epoch_acc_valance += pred_acc_valance
            pred_epoch_acc_arousal += pred_acc_arousal
            pred_epoch_pred_f1 += pred_f1

            # max returns (value ,index)
            _, y_pred_tags = torch.max(y_pred.data, 1)
            y_pred_list.append(y_pred_tags.detach().cpu().numpy())
            y_true_list.append(y_true.detach().cpu().numpy())

        pred_epoch_acc /= len(loader)
        pred_epoch_acc_valance /= len(loader)
        pred_epoch_acc_arousal /= len(loader)
        pred_epoch_pred_f1 /= len(loader)
        y_pred_list = np.array(y_pred_list).flatten()
        y_true_list = np.array(y_true_list).flatten()
        return y_pred_list, y_true_list, pred_epoch_acc, pred_epoch_acc_valance, pred_epoch_acc_arousal, pred_epoch_pred_f1


def plot_confusion_matrix(cm, classes, save=False, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("")
    else:
        print('Confusion matrix, without normalization')

    # print(cm) # uncomment if I you want confusion matrix in terminal
    plt.figure(title)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if save:
        wandb.log({title: plt})
    else:
        plt.show()
