import torch
from src.autoencoder import AudioAutoencoder, load_wav2vec_model
from src.data_loader import load_librispeech
import torchaudio

# Carregar o modelo treinado
wav2vec_model = load_wav2vec_model('./models/wav2vec/wav2vec.pt')
autoencoder = AudioAutoencoder(wav2vec_model)
autoencoder.load_state_dict(torch.load('./models/checkpoints/autoencoder_epoch_10.pth'))
autoencoder.eval()

# Carregar dados de teste
data_loader = load_librispeech(split='test-clean')

# Avaliar e salvar saídas
for i, batch in enumerate(data_loader):
    audio, _, _ = batch
    with torch.no_grad():
        reconstructed = autoencoder(audio)
    
    # Salvar áudio reconstruído
    torchaudio.save(f'./outputs/reconstructions/reconstructed_{i}.wav', reconstructed.cpu(), sample_rate=16000)
