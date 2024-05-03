import load_data as data
import matplotlib.pyplot as plt 

rdt = data.rdt
dt = data.dt


# Plotting the first 9 frames of RDT data
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

for i, ax in enumerate(axes.flatten()):
    ax.imshow(rdt[0][i, :, :], cmap='jet', aspect='auto')
    ax.set_title(f'Frame {i}')
    ax.set_xlabel('Doppler')
    ax.set_ylabel('Range')
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()