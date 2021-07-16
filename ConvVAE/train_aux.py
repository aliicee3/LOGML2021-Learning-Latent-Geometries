import torch
from model import vae_loss
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

def get_batch(data, batch_size=64):
    total_len = data.shape[0]
    
    index = np.arange(total_len)
    index = np.random.permutation(index)

    for i in range(0, total_len, batch_size):
        b_index = index[i:min(i+batch_size,total_len)]
        yield data[b_index,:,:,:]

def plot_gallery(images, n_row=3, n_col=6, with_title=False, titles=[]):
    plt.figure(figsize=(1.5 * n_col, 1.7 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        try:
            plt.imshow(images[i].transpose(1, 2, 0), vmin=0, vmax=1, interpolation='nearest')
            if with_title:
                plt.title(titles[i])
            plt.xticks(())
            plt.yticks(())
        except:
            pass
    plt.pause(0.05)

import cv2

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return np.float32(cv2.LUT(image.astype('uint8'), table))

def fit_epoch_vae(model, train_x, optimizer, batch_size, is_cnn=False, device = 'cuda:0'):
    running_loss = 0.0
    processed_data = 0
    model.train()
    
    for inputs in get_batch(train_x, batch_size):
        # Little data augmentation
        coin = np.random.uniform(0, 1)
        # Horizontal flip
        if coin > 0.5: 
            inputs = inputs[:,:,:,::-1]
        
        light = np.random.uniform(0.85,1.15)
        inputs = adjust_gamma(inputs*255, light)/255        
        
        inputs = torch.tensor(inputs)
        inputs = inputs.to(device)
        # Forward Pass
        reconstructed_x, mu, log_var = model.forward(inputs)
        loss = vae_loss(reconstructed_x, inputs, mu, log_var, 1e-4)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        
        # Apply Jacobians to Weights
        optimizer.step()
        
        running_loss += loss.item() * inputs.shape[0]
        processed_data += inputs.shape[0]
    
    train_loss = running_loss / processed_data    
    return train_loss

def train_vae(train_x, model, epochs=10, batch_size=32, lr=0.001, device = 'cuda:0'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)        
    history = []
    log_template = "\nEpoch {ep:03d} train_loss: {t_loss:0.4f}"
    
    best = 1e12

    with tqdm(desc="epoch", total=epochs) as pbar_outer:
        for epoch in range(epochs):            
            train_loss = fit_epoch_vae(model,train_x,optimizer,batch_size, device = device)

            history.append((train_loss))

            pbar_outer.update(1)
            tqdm.write(log_template.format(ep=epoch+1, t_loss=train_loss))
            if epoch % 50 == 0:
                torch.save(model.state_dict(), "weights/%s_Model_b.pt" %epoch)
        
    
    torch.save(model.state_dict(), "weights/finalModel_b.pt")
    return history