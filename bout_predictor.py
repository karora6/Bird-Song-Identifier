import numpy as np
import pandas as pd
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view
import torch
from torchvision import transforms
import sys
sys.path.append('/mnt/cube/k5arora/code/')
from ml_tools import cvgg1

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=(320, 320)),
])
DEVICE = torch.device('cpu')
MODEL = cvgg1.CustomVGG1(input_shape=1, 
                 hidden_units=20, 
                 output_shape=3).to(DEVICE)
MODEL_PATH = '/mnt/cube/k5arora/proj/state_dicts/CustomVGG1.pickle'
MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))

CLASS_NAMES = {0: "Noise", 1: "Call", 2: "Song"}


def chunk_row(row, chunk_size, cutoff_size, overlap_percent, ms_per_bin=5):
    spec = row['spectrogram']
    total_spec_length = spec.shape[1]
    len_ms = total_spec_length * ms_per_bin
    waveform = row['waveform']
    spec_num = row.name
    
    step = int(chunk_size * (1 - overlap_percent))
    if total_spec_length > chunk_size:
        windows = sliding_window_view(spec, (spec.shape[0], chunk_size), (0,1)).squeeze()
        chunks = windows[::step]
        if (total_spec_length - chunk_size) % step >= cutoff_size:
            final_chunks = np.concatenate((chunks, [windows[-1]]), axis=0)
        else:
            final_chunks = chunks
    elif total_spec_length >= cutoff_size:
        final_chunks = np.array([pad_spec_with_zeros(spec, chunk_size)])

    chunk_size_as_percent = chunk_size / total_spec_length
    wf_chunk = chunk_size_as_percent * waveform.shape[0]
    wf_step = wf_chunk * (1 - overlap_percent)
    if total_spec_length > chunk_size:
        wf_windows = sliding_window_view(waveform, int(wf_chunk), 0).squeeze()
        waveforms = wf_windows[::int(wf_step)]
        if (total_spec_length - chunk_size) % step >= cutoff_size:
            final_waveforms = np.concatenate((waveforms, [wf_windows[-1]]), axis=0)
        else:
            final_waveforms = waveforms
    elif total_spec_length >= cutoff_size:
        final_waveforms = np.array([waveform])

    time_chunk = ms_per_bin * chunk_size
    time_step = time_chunk * (1 - overlap_percent)
    if total_spec_length > chunk_size:
        time_windows = sliding_window_view(np.arange(len_ms), int(time_chunk), 0).squeeze()
        len_ms_ranges = time_windows[::int(time_step)]
        cutoff_size_percent = cutoff_size / total_spec_length
        if (total_spec_length - chunk_size) % step >= cutoff_size:
            final_len_ms_ranges = np.concatenate((len_ms_ranges, [time_windows[-1]]), axis=0)
        else:
            final_len_ms_ranges = len_ms_ranges
    elif total_spec_length >= cutoff_size:
        final_len_ms_ranges = np.array([np.arange(len_ms)])
    
    processed_rows = [(final_chunks[i], final_waveforms[i], final_len_ms_ranges[i][-1] - final_len_ms_ranges[i][0], spec_num, i, len(final_chunks)) for i in range(final_chunks.shape[0])]
    return processed_rows


def get_pred_col_chunks(spec, model, transform, class_names):
    img = Image.fromarray(spec)
    transformed_img = transform(img)
    
    model.eval()
    with torch.inference_mode():
        pred_logits = model(transformed_img.unsqueeze(dim=0).to(DEVICE))

    pred_probs = torch.softmax(pred_logits, dim=1)
    pred_label = torch.argmax(pred_probs, dim=1)
    
    prob = float(f'{pred_probs.max().cpu():.3f}')
    label = class_names[pred_label.item()]

    return pd.Series([prob, label])


def add_pred_col_chunks(cdf):
    cdf[['confidence', 'prediction']] = cdf['spectrogram'].apply(lambda spec: get_pred_col_chunks(spec, 
                                                                                                  model=MODEL, 
                                                                                                  transform=TRANSFORM, 
                                                                                                  class_names=CLASS_NAMES))
    return cdf
    
    
def determine_spec_label(chunk_labels):
    if sum(chunk_labels == 'Song') >= 2:
        return 'Song'
    elif sum(chunk_labels == 'Call') >= 2:
        return 'Call'
    else:
        return 'Noise'

    
def chunk(df): 
    processed_df = df.apply(lambda row: chunk_row(row, 
                                                  chunk_size=400, 
                                                  cutoff_size=200, 
                                                  overlap_percent=0.5), axis=1)
    flattened_df = [item for sublist in processed_df for item in sublist]
    chunked_df = pd.DataFrame(flattened_df, columns=['spectrogram', 'waveform', 'len_ms', 'spec_num', 'index', 'chunk_num'])
    return chunked_df


def add_pred_col(df, cdf):
    spec_labels = cdf.groupby(['spec_num'])['prediction'].apply(determine_spec_label).reset_index()
    spec_labels.columns = ['spec_num', 'spec_label']
    spec_labels['confidence'] = cdf.groupby(['spec_num'])['confidence'].mean()
    spec_labels.sort_values('spec_num', ascending=False)
    df['prediction'] = spec_labels['spec_label']
    df['confidence'] = spec_labels['confidence']
    df['bout_check'] = np.where(df['prediction'] == 'Song', True, False)
    df['is_call'] = np.where(df['prediction'] == 'Call', True, False)
    df = df.sort_values('confidence', ascending=False)
    return df


def predict(df):
    cdf = chunk(df)
    cdf = add_pred_col_chunks(cdf)
    return add_pred_col(df, cdf)

