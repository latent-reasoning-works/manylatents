import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from data_loader import load_data_1000G, load_data_HGDP
import os
from preprocess import maf_scale

# Define the Autoencoder model
class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=100):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, latent_dim)  # Latent space of size 100
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, input_dim),
            nn.Sigmoid()  # Output between 0 and 1 (assuming normalized input)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

# Function to add noise to the data (optional)
def add_noise(data, noise_factor=0.3):
    noisy_data = data + noise_factor * torch.randn_like(data)
    noisy_data = torch.clamp(noisy_data, 0., 1.)  # Clip the data to be within [0, 1]
    return noisy_data

# Training function
def train_autoencoder(model, dataloader, criterion, optimizer, num_epochs=20, device='cpu', add_noise_flag=True, noise_factor=0.3, model_save_path='autoencoder.pth'):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in dataloader:
            inputs = batch[0].to(device)  # Move data to the correct device
            
            # Add noise if the flag is True
            if add_noise_flag:
                inputs_noisy = add_noise(inputs, noise_factor)
            else:
                inputs_noisy = inputs

            # Forward pass
            optimizer.zero_grad()
            reconstructed, latent = model(inputs_noisy)
            loss = criterion(reconstructed, inputs)  # Reconstruction loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        # Print training stats for the epoch
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        
        # Save the model at the end of each epoch
        torch.save(model.state_dict(), model_save_path)

    print(f"Model saved to {model_save_path} after {num_epochs} epochs.")

# Main function to set up the data and run the training
def main(data, input_dim, latent_dim=100, num_epochs=20, batch_size=64, learning_rate=1e-3, add_noise_flag=True, noise_factor=0.3, model_save_path='autoencoder.pth'):
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare the dataset and dataloader
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the autoencoder model
    model = DenoisingAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)  # Move model to the correct device

    # Define the loss function and the optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-7)

    # Train the autoencoder
    train_autoencoder(model, dataloader, criterion, optimizer, num_epochs=num_epochs, device=device, add_noise_flag=add_noise_flag, noise_factor=noise_factor, model_save_path=model_save_path)

    # After training, return the trained model for downstream analysis
    return model

# Example usage:
# Assuming `data` is a PyTorch tensor of your input data (e.g., torch.Tensor(data)) and input_dim is the number of features in the input data.
if __name__ == "__main__":
    
    data_name = 'HGDP'   # or '1KGP'
    # define paths
    PROJECT_PATH = '/home/mila/s/shuang.ni/phate_genetics/shuang/'
    SCRATCH_PATH = '/home/mila/s/shuang.ni/scratch/phate_genetics/'
    SAVE_PATH = SCRATCH_PATH + 'results/'
    Figure_PATH = PROJECT_PATH + 'figures/'

    # import data
    # load data from original
    if data_name == '1KGP':    
        DATA_PATH = SCRATCH_PATH + '1KGP/V3/'
        fname = '1000G.2504_WGS30x.GSA17k_MHI.intersectGSA.miss10perc.maf0.05.pruned.autosomes.noHLA.phased_imputed.hdf5'
        data, class_labels, _, _, _ = load_data_1000G(os.path.join(DATA_PATH, fname))
    elif data_name == 'HGDP':
        DATA_PATH = SCRATCH_PATH + 'HGDP+1KGP/V4/'
        _, _, data, _= load_data_HGDP(DATA_PATH)
        
    maf_scaled_data = maf_scale(data)
    data_tensor = torch.from_numpy(maf_scaled_data).float()

    input_dim = data.shape[1]  # Get the number of features from your data
    model_save_path = SAVE_PATH + 'AE/ae_trained_model_2D_' + data_name + '.pth'
    trained_model = main(data_tensor, input_dim=input_dim, latent_dim=2, num_epochs=200, batch_size=128, learning_rate=5e-4, add_noise_flag=False, noise_factor=0.01, model_save_path=model_save_path)

