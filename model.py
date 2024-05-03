import tensorflow as tf
from keras.layers import Input,Dropout,Reshape,Permute,Conv1D,Conv3D, Permute,Flatten,Dense
from keras.models import Model,Sequential

model = Sequential()


def create_model(input_shape):
    # Input layer
    inputs = Input(shape=input_shape)

    # Shape variant layers
    x = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 2, 1), padding='same', activation='relu')(inputs)
    x = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(2, 2, 1), padding='same', activation='relu')(x)
    x = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(2, 2, 1), padding='same', activation='relu')(x)
    x = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(2, 2, 1), padding='same', activation='relu')(x)

    # Shape invariant layers
    x = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')(x)
    x = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')(x)
    x = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')(x)
    x = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')(x)
    x = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')(x)

    # Flatten layer
    # Flatten layer
    x = Flatten()(x)
    
    # Reshape layer
    x = Reshape((512, 1000))(x)
    # Additional layers
    x = Permute((2, 1))(x)

    x = Conv1D(filters=3, kernel_size=1)(x)

    # Assuming you have a custom SelfAttention layer
   # x = SelfAttention(4,32)(x)

    # Additional layers
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = Dense(3, activation='softmax')(x)  # Assuming 3 output classes

    # Define model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Input shape
input_shape = (32, 64, 1000, 1)  # Assuming range x doppler x time x channels

# Create the model
model = create_model(input_shape)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
# model.summary()
