# Numpy and pandas
import pandas as pd 
import numpy as np

# Torch packages
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler

# Other packages
import os
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import yaml



class Dataset():
    def __init__(self, parameters):
        # Loading path
        self._path_load = parameters['Path']['load']
        self.fileType = ".csv"

        # Loading labels
        self.TARGET_LABELS = parameters['Labels']['target']
        self.GAME_LABELS = parameters['Labels']['game']
        self.OPENFACE_LABELS =  parameters['Labels']['openface']
        self.OPENPOSE_LABELS = parameters['Labels']['openpose']
        self.OPENSMILE_LABELS = parameters['Labels']['opensmile']

        # Loading number of classes in labels
        self.opensmile_window = parameters['Settings']['opensmile_window']

    def load_data(self):
        print("--- Loading train and target-files...")
        self.x_face = pd.read_csv(self._path_load + "openface.csv") 
        print("\t--- Load face-file")
        self.x_pose = pd.read_csv(self._path_load + "openpose.csv") 
        print("\t--- Load pose-file")
        self.x_smile = pd.read_csv(self._path_load + "opensmile_" + self.opensmile_window + ".csv") 
        print("\t--- Load smile-file")
        self.x_game = pd.read_csv(self._path_load + "gamestate.csv")
        print("\t--- Load game_rec-file")
        self.x_game_rec  = pd.read_csv(self._path_load + "gamestate_rec.csv")
        print("\t--- Load game-file")
        self.y_target = pd.read_csv(self._path_load + "target.csv") 
        print("\t--- Load target-file")
 
    def _split(self, ratio, y_target):
        print("--- Using sratified split")
        dataset_size = len(y_target)
        indices = list(range(dataset_size))

        # Get the split ratio
        train_split = (ratio[1]+ratio[2])/100
        valid_test_split = 1/(ratio[1]+ratio[2])*10
        # Get indices for each split
        train_indices, indices, y_train, y_valid_test = train_test_split(indices, y_target, test_size=train_split, random_state=42, stratify=y_target)
        valid_indices, test_indices, y_valid, y_test = train_test_split(indices, y_valid_test, test_size=valid_test_split, random_state=42, stratify=y_valid_test)

        print("\tTraining class split: ", np.unique(y_train, return_counts=True))
        print("\tValidating class split: ", np.unique(y_valid, return_counts=True))
        print("\tTesting class split: ", np.unique(y_test, return_counts=True))
        print(f"\ttrain: {len(train_indices)}, valid: {len(valid_indices)}, test: {len(test_indices)}")
        return train_indices, valid_indices, test_indices

    def _normalize(self, x_train, x_valid, x_test, dim, label_dim, normalize):
        if normalize=="MinMax":
            scaler= preprocessing.MinMaxScaler()
        elif normalize=="Standard":
            scaler= preprocessing.MinMaxScaler()
        else:
            print(f"\n--- Could not find normalize option matching: {normalize} \n\tWill use original values.\n")
            return x_train, x_valid, x_test
        x_train = np.reshape(scaler.fit_transform(x_train.reshape(-1, label_dim)), dim)
        x_valid = np.reshape(scaler.transform(x_valid.reshape(-1, label_dim)), dim)
        x_test = np.reshape(scaler.transform(x_test.reshape(-1, label_dim)), dim)
        return x_train, x_valid, x_test

    def preprocess_recurrent(self, batch_size=32, normalize="MinMax", split=(70,20,10), shuffle=True, stratified=True, remove=False):   
        x_face = self.x_face[self.OPENFACE_LABELS].to_numpy(dtype=float)
        x_pose = self.x_pose[self.OPENPOSE_LABELS].to_numpy(dtype=float)
        x_smile = self.x_smile[self.OPENSMILE_LABELS].to_numpy(dtype=float)
        x_game_rec = self.x_game_rec[self.GAME_LABELS].to_numpy(dtype=float)
        y_target = self.y_target[self.TARGET_LABELS].to_numpy(dtype=float)
        face_conf = self.x_face[' confidence']
        
        
        # reshape with seq-len
        face_dim = (-1, 50, x_face.shape[1])
        pose_dim = (-1, 50, x_pose.shape[1])
        smile_seq = round((2-2/int(self.opensmile_window))/0.04 + 1) # Smile seq-len
        smile_dim = (-1, smile_seq, x_smile.shape[1])
        game_rec_seq = 594
        game_dim = (-1, game_rec_seq, x_game_rec.shape[1])

        # Reshape with seq-len
        x_face = np.reshape(x_face, face_dim)
        x_pose = np.reshape(x_pose, pose_dim)
        x_smile = np.reshape(x_smile, smile_dim)
        x_game_rec = np.reshape(x_game_rec, game_dim)
        print(f"\tx_face: {x_face.shape}, x_pose: {x_pose.shape}, x_smile: {x_smile.shape}, x_game_rec: {x_game_rec.shape}\n")

        # Remove bad data
        if remove:
            remove_index = list()
            for index in range(x_face.shape[0]):
                count_bad_data = 0
                for seq_len in range(50):
                    if face_conf[index + seq_len] < 0.75:
                        count_bad_data += 1
                if count_bad_data >= 25:
                    remove_index.append(index)
            x_face = np.delete(x_face, remove_index, axis=0)
            x_pose = np.delete(x_pose, remove_index, axis=0)
            x_smile = np.delete(x_smile, remove_index, axis=0)
            x_game = np.delete(x_game_rec, remove_index, axis=0)
            y_target = np.delete(y_target, remove_index, axis=0)

            print(f'--- Removed: {len(remove_index)}, unvalid samples')
            print(f"\tx_face: {x_face.shape}, x_pose: {x_pose.shape}, x_smile: {x_smile.shape}, x_game: {x_game.shape}\n")

        # Dataset split -> train, validation, test data
        train_indices, valid_indices, test_indices = self._split(split, y_target)
        x_face_train = x_face[tuple([train_indices])]
        x_pose_train = x_pose[tuple([train_indices])]
        x_smile_train = x_smile[tuple([train_indices])]
        x_game_train = x_game[tuple([train_indices])]
        y_train = y_target[tuple([train_indices])]

        x_face_valid = x_face[tuple([valid_indices])]
        x_pose_valid = x_pose[tuple([valid_indices])]
        x_smile_valid = x_smile[tuple([valid_indices])]
        x_game_valid = x_game[tuple([valid_indices])]
        y_valid = y_target[tuple([valid_indices])]

        x_face_test = x_face[tuple([test_indices])]
        x_pose_test = x_pose[tuple([test_indices])]
        x_smile_test = x_smile[tuple([test_indices])]
        x_game_test = x_game[tuple([test_indices])]
        y_test = y_target[tuple([test_indices])]

        # Normalize data
        print(f'\n--- normalizer: {normalize}')
        x_face_train, x_face_valid, x_face_test = self._normalize(x_face_train, x_face_valid, x_face_test, face_dim, len(self.OPENFACE_LABELS), normalize)
        x_pose_train, x_pose_valid, x_pose_test = self._normalize(x_pose_train, x_pose_valid, x_pose_test, pose_dim, len(self.OPENPOSE_LABELS), normalize)
        x_smile_train, x_smile_valid, x_smile_test = self._normalize(x_smile_train, x_smile_valid, x_smile_test, smile_dim, len(self.OPENSMILE_LABELS), normalize)
        x_game_train, x_game_valid, x_game_test = self._normalize(x_game_train, x_game_valid, x_game_test, game_dim, len(self.GAME_LABELS), normalize)
        
        # numpy -> tensor
        x_face_train, x_face_valid, x_face_test = torch.Tensor(x_face_train).float(), torch.Tensor(x_face_valid).float(), torch.Tensor(x_face_test).float()
        x_pose_train, x_pose_valid, x_pose_test = torch.Tensor(x_pose_train).float(), torch.Tensor(x_pose_valid).float(), torch.Tensor(x_pose_test).float()
        x_smile_train, x_smile_valid, x_smile_test = torch.Tensor(x_smile_train).float(), torch.Tensor(x_smile_valid).float(), torch.Tensor(x_smile_test).float()
        x_game_train, x_game_valid, x_game_test = torch.Tensor(x_game_train).float(), torch.Tensor(x_game_valid).float(), torch.Tensor(x_game_test).float()
        
        # Flatten target & numpy -> tensor
        y_train, y_valid, y_test = y_train.flatten(), y_valid.flatten(), y_test.flatten()
        y_train = torch.Tensor(y_train).long()
        y_valid = torch.Tensor(y_valid).long() 
        y_test = torch.Tensor(y_test).long() 
        
        # Datasets
        train_dataset = TensorDataset(x_face_train, x_pose_train, x_smile_train, x_game_train, y_train)
        valid_dataset = TensorDataset(x_face_valid, x_pose_valid, x_smile_valid, x_game_valid, y_valid)
        test_dataset = TensorDataset(x_face_test, x_pose_test, x_smile_test, x_game_test, y_test)

        # Weighted Sampling
        class_sample_count = np.unique(y_train, return_counts=True)[1]
        weight = 1.0 / torch.tensor(class_sample_count, dtype=torch.float)
        print(f'\n--- Class weights: {weight}\n')

        # Dataloaders
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=1000)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1000)
        
        return train_loader, valid_loader, test_loader, weight


if __name__ == '__main__':
    # load params
    with open("parameters/params.yml", "r") as f:
        hypes = yaml.safe_load(f)

    data = Dataset(parameters=hypes['Dataset'])
    data.load_data()
    data.preprocess_recurrent(remove=True)
    