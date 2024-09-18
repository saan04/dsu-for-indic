import torch
import torchaudio
import torchaudio.transforms as T
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import os
from tqdm import tqdm
import multiprocessing as mp

mp.set_start_method('spawn', force=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the wav2vec2 model
bundle = torchaudio.pipelines.WAV2VEC2_XLSR53
model = bundle.get_model().to(device)

def extract_wav2vec2_features(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != bundle.sample_rate:
        resampler = T.Resample(sample_rate, bundle.sample_rate)
        waveform = resampler(waveform)
    
    waveform = waveform.to(device)
    
    with torch.no_grad():
        features, _ = model.extract_features(waveform)
    
    last_layer_features = features[-1].squeeze(0).transpose(0, 1)
    return last_layer_features.cpu().numpy()

def get_discrete_units(features, n_clusters=100):
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=1000)
    discrete_units = kmeans.fit_predict(features)
    return discrete_units

def process_file(file_path, n_clusters=100):
    try:
        features = extract_wav2vec2_features(file_path)
        discrete_units = get_discrete_units(features, n_clusters)
        output_string = ' '.join(map(str, discrete_units))
        return file_path.split('/')[-1], output_string
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def process_files(file_list, n_clusters):
    results = []
    for file in tqdm(file_list, desc="Processing files"):
        result = process_file(file, n_clusters)
        if result:
            results.append(result)
    return results

def process_directory(input_dir, output_file, n_clusters=100, num_processes=4):
    wav_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))

    chunks = np.array_split(wav_files, num_processes)
    
    with mp.Pool(processes=num_processes) as pool:
        all_results = pool.starmap(process_files, [(chunk.tolist(), n_clusters) for chunk in chunks])
    
    results = [item for sublist in all_results for item in sublist]

    with open(output_file, 'w') as f:
        for file_name, output_string in results:
            f.write(f"{file_name}|{output_string}\n")

input_directory = #MUST-C dataset format preferred 
output_file = #output text file
n_clusters = 100  #k-means clustering based on number - 50,100,150
num_processes = 4  #based on cores

if __name__ == '__main__':
    process_directory(input_directory, output_file, n_clusters, num_processes)
    print(f"Processing completed. Results saved to {output_file}")