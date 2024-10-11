import torch
import torch.optim as optim
import torch.nn as nn
from src.autoencoder.py import AudioAutoencoder, load_wav2vec_model
from src.data_loader import load_librispeech

# Definir dispositivos
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Carregar Wav2Vec
wav2vec_model = load_wav2vec_model('./models/wav2vec/wav2vec.pt')
wav2vec_model.to(device)

# Inicializar o Autoencoder
autoencoder = AudioAutoencoder(wav2vec_model)
autoencoder.to(device)

# Configurações de treinamento
optimizer = optim.Adam(autoencoder.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

# Carregar dados
data_loader = load_librispeech()

# Treinar
num_epochs = 10
for epoch in range(num_epochs):
    autoencoder.train()
    for batch in data_loader:
        audio, _, _ = batch
        audio = audio.to(device)
        
        # Forward pass
        reconstructed = autoencoder(audio)
        loss = loss_fn(reconstructed, audio)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Salvar checkpoint
torch.save(autoencoder.state_dict(), './models/checkpoints/autoencoder_epoch_{}.pth'.format(epoch+1))
