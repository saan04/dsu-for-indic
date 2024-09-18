import numpy as np
import torch
from transformers import HubertModel
import os
import soundfile as sf
from tqdm import tqdm


class SimpleKMeans:
    def __init__(self, n_clusters=100, max_iter=300, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None

    def fit(self, X):
        np.random.seed(self.random_state)
        idx = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[idx]

        for _ in range(self.max_iter):
            old_centroids = self.centroids.copy()
            labels = self.predict(X)
            for i in range(self.n_clusters):
                if np.sum(labels == i) > 0:
                    self.centroids[i] = np.mean(X[labels == i], axis=0)
            if np.all(old_centroids == self.centroids):
                break

    def predict(self, X):
        distances = np.sqrt(((X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)

class KMeansIndex:
    def __init__(self, n_clusters=100):
        self.kmeans = SimpleKMeans(n_clusters=n_clusters)
        self.fitted = False

    def fit(self, X):
        self.kmeans.fit(X)
        self.fitted = True

    def predict(self, X):
        if not self.fitted:
            raise ValueError("KMeans model hasn't been fitted yet.")
        return self.kmeans.predict(X)

def load_or_create_index(index_path, hidden_states_dim):
    if os.path.exists(index_path):
        centroids = np.load(index_path)
        kmeans_index = KMeansIndex()
        kmeans_index.kmeans.centroids = centroids
        kmeans_index.fitted = True
    else:
        kmeans_index = KMeansIndex()
    return kmeans_index

def get_centroids_index(xq, index):
    return index.predict(xq)

class Hubert2Unit(torch.nn.Module):
    def __init__(
        self,
        model_name="utter-project/mHuBERT-147-base-2nd-iter",
        index_path="mhubert147_kmeans_centroids.npy",
        dtype=torch.float32,
        device="cuda:0",
    ):
        super(Hubert2Unit, self).__init__()
        self.model = HubertModel.from_pretrained(model_name).eval()
        self.model.to(dtype=dtype, device=device)
        hidden_states_dim = self.model.config.hidden_size
        self.index = load_or_create_index(index_path, hidden_states_dim)

    def zero_mean_unit_var_norm(
        self, input_values, wav_lengths, padding_value: float = 0.0
    ):
        if wav_lengths is not None:
            normed_input_values = []
            for vector, length in zip(input_values, wav_lengths):
                normed_slice = (vector - vector[:length].mean()) / torch.sqrt(vector[:length].var() + 1e-7)
                if length < normed_slice.shape[0]:
                    normed_slice[length:] = padding_value
                normed_input_values.append(normed_slice)
        else:
            normed_input_values = [(x - x.mean()) / torch.sqrt(x.var() + 1e-7) for x in input_values]
        return torch.stack(normed_input_values, dim=0)

    def forward(self, wav, wav_lengths, do_normalize=True):
        with torch.no_grad():
            if do_normalize:
                input_values = self.zero_mean_unit_var_norm(wav, wav_lengths)
            else:
                input_values = wav.clone()
            attention_mask = torch.arange(
                input_values.size(1), 
                device=input_values.device)[None, :] < wav_lengths[:, None]
            attention_mask = attention_mask.long()
            hidden_states = self.model(
                input_values, 
                attention_mask=attention_mask,
                output_hidden_states=True
            ).hidden_states[9]  # 9th layer of encoder block
            hidden_states = hidden_states.reshape(hidden_states.size(0) * hidden_states.size(1), -1)
            hidden_states_cpu = hidden_states.float().detach().cpu().numpy()
            
            if not self.index.fitted:
                self.index.fit(hidden_states_cpu)
                np.save("mhubert147_kmeans_centroids.npy", self.index.kmeans.centroids)
                print("KMeans model fitted and saved.")
            
            C = get_centroids_index(hidden_states_cpu, self.index)
            C = C.reshape(wav.shape[0], -1)
        return C

def process_wav_directory(input_dir, output_file):
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    hubert = Hubert2Unit(device=device)
    
    wav_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    
    with open(output_file, 'w') as f:
        for wav_file in tqdm(wav_files, desc="Processing WAV files"):
            wav_path = os.path.join(input_dir, wav_file)
            wav, sample_rate = sf.read(wav_path)
            
            wav = torch.tensor(wav).to(device).unsqueeze(0).float()
            lengths = torch.tensor([wav.shape[1]]).to(device)
            
            C = hubert(wav, lengths)
            
            dsu_string = ' '.join(map(str, C.flatten()))
            f.write(f"{wav_file}|{dsu_string}\n")
    
    print(f"Processed {len(wav_files)} WAV files. Results saved in {output_file}")
    
lst = ['Malayalam', 'Tamil', 'Telugu']
for i in lst:
    input_directory = 
    output_directory = 
    process_wav_directory(input_directory, output_directory)