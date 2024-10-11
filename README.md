# Projeto de Autoencoder com Wav2Vec e LibriSpeech

Este projeto visa construir um autoencoder para reconstruir amostras de áudio utilizando o Wav2Vec da Meta (Fairseq) e o dataset LibriSpeech.

## Estrutura do projeto

- `data/`: Contém o dataset LibriSpeech.
- `models/`: Contém os modelos pré-treinados e os checkpoints do autoencoder.
- `src/`: Código-fonte para carregar dados, treinar e avaliar o autoencoder.
- `outputs/`: Logs e áudios reconstruídos.
- `notebooks/`: Notebooks para análises exploratórias.

## Como usar

### Dependências

Instale as dependências utilizando o `requirements.txt`:

```bash
pip install -r requirements.txt


