import torch
import torch.nn as nn
from fairseq.models.wav2vec import Wav2VecModel

class AudioAutoencoder(nn.Module):
    def __init__(self, wav2vec_model):
        super(AudioAutoencoder, self).__init__()
        self.encoder = wav2vec_model  # Usar Wav2Vec como encoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 1, kernel_size=4, stride=2),
            nn.Tanh()
        )
    
    def forward(self, audio):
        latent = self.encoder(audio)['x']  # Extrai o embedding do wav2vec
        reconstructed = self.decoder(latent)
        return reconstructed

def load_wav2vec_model(model_path):
    model, cfg, task = Wav2VecModel.from_pretrained(model_path)
    model.eval()
    return model
