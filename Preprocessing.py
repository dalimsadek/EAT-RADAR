import load_data as data
import config as c

rdt = data.rdt
dt = data.dt
labels = data.labels
id = data.id

# Preprocessing for X

frame_shape = c.frame_shape


s = 0
for i in rdt :
    s+=i.shape[0]
num_frames_to_add = 1000 - (num_frames_to_add % 1000)

null_frames = np.zeros((num_frames_to_add,) + frame_shape)
    

concatenated_array = np.concatenate(rdt, axis=0)
extended_array = np.concatenate([concatenated_array, null_frames], axis=0)

X = extended_array.reshape((-1, 32, 64, 1000, 1))

# Preprocessing for y 

label_for_id = list(labels.iloc[0, id])
label_for_id = label_for_id + [0] * num_frames_to_add

y = np.array(label_for_id)
y_reshaped = np.array(y).reshape(-1, 1000)
num_classes = 3  # Assuming you have 3 classes
y_one_hot = np.eye(num_classes)[y_reshaped]

# Reshape again to match the expected shape of y for the model
y_final = y_one_hot.reshape(-1, 1000, num_classes)

y = y_final 




