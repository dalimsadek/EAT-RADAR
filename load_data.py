import pickle
import pandas as pd
import numpy as np
import config as c


id = c.id
folder_path = c.folder_path 


def read_pkl_to_dataframe(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return pd.DataFrame(data)


def load_rdt_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

dt_dataframes = [read_pkl_to_dataframe(f'Data/Doppler-Time Data/DT_{i}.pkl') for i in range(70)]
rdt_dataframes = [load_rdt_data(f'Data/RDT/RDT_{i}.pkl') for i in range(id,id+1)]

with open('Data/Labels/data_Y_70.pkl', 'rb') as f:
    data = pickle.load(f)

# Create a DataFrame with meal ID index
labels_df = pd.DataFrame({'Labels': data}, index=range(70))

# 4. 7-fold cross validation id_list file
id_list_data = {}
for i in range(7):
    folder_name = str(i)
    train_ids = np.load(f'Data/7-fold id_list/{folder_name}/train.npy')
    test_ids = np.load(f'Data/7-fold id_list/{folder_name}/test.npy')
    valid_ids = np.load(f'Data/7-fold id_list/{folder_name}/valid.npy')
    
    id_list_data[folder_name] = {
        'train_ids': train_ids,
        'test_ids': test_ids,
        'valid_ids': valid_ids
    }

for i in range(len(dt_dataframes)) : 
    dt_dataframes[i] = pd.DataFrame(dt_dataframes[i])


rdt = rdt_dataframes
dt = dt_dataframes
labels = labels_df
cross_val_ids = id_list_data
