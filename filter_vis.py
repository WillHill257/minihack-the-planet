import os
from natsort import natsorted
import torch
import matplotlib.pyplot as plt


# get the name of the latest model
model_name = natsorted(os.listdir('./models'))[-1]

print(f'Loading model {model_name}')

checkpoint = torch.load('./models/' + model_name)
model_state = checkpoint['model_state_dict']

# get the weights of the first conv layer
conv1_weights = model_state['conv_crop1.weight']

# plot the weights of the first conv layer
for i, filter in enumerate(conv1_weights):
    # put filter on to the cpu
    filter = filter.cpu()
    # make it a tight layout
    plt.subplot(4,4,i+1)
    plt.imshow(filter[0,:,:], cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.show()