import argparse
from backend_logic.combine import combine_quadro
import librosa
import scipy.ndimage
import torch

from PretrainedSED.data_util import audioset_classes
from PretrainedSED.helpers.decode import batched_decode_preds
from PretrainedSED.helpers.encode import ManyHotEncoder
from PretrainedSED.models.frame_mn.Frame_MN_wrapper import FrameMNWrapper
from PretrainedSED.models.prediction_wrapper import PredictionsWrapper
from PretrainedSED.models.frame_mn.utils import NAME_TO_WIDTH
import numpy as np

config = {
    "model_name": "frame_mn06",
    "detection_thresholds": (0.1, 0.2, 0.5),
    "median_window": 9,
    "threshold": 0.01,  # detection threshold for events, low pass as we combine multiple channels
    "chunk_duration": 0.96,  # approximate duration of each chunk in seconds
}

# params from pretrained SED

def get_model(model_name):
    if not model_name.startswith("frame_mn"):
        raise ValueError("Only frame_mn models are supported in this script.")
    
    width = NAME_TO_WIDTH(model_name)
    frame_mn = FrameMNWrapper(width)
    embed_dim = frame_mn.state_dict()['frame_mn.features.16.1.bias'].shape[0]
    model = PredictionsWrapper(frame_mn, checkpoint=f"{model_name}_strong_1", embed_dim=embed_dim)
    return model


model = get_model(config["model_name"])

def predict_events_SED(audio_chunks, sample_rates, channels_audios, skip_model=False):
    channels = channels_audios[0] # assume all have same channels, same audio devic
    sample_rate = sample_rates[0] # assume all have same sample rate, same audio device
    audio_chunk = np.concatenate(audio_chunks, axis=0) # shape [total_samples, channels]
    amount_chunks = len(audio_chunks)
     
    if model is None or audio_chunk is None:
        return [{"event": "model_not_loaded", "confidence": 0.0, "orientation": "NAN"}]
    
    if isinstance(audio_chunk, torch.Tensor):
        audio_tensor = np.ascontiguousarray(audio_chunk.detach().cpu().numpy(), dtype=np.float32)
    elif isinstance(audio_chunk, np.ndarray):  
        audio_tensor = np.ascontiguousarray(audio_chunk, dtype=np.float32)
    else:
        raise TypeError("audio_chunk must be a numpy array or torch Tensor")

    if channels == 2:
        pass
    elif channels == 4:
        return process_predict_quadro_SED(audio_tensor, sample_rate, skip_model,amount_chunks)
    elif channels == 6: # 5.1 = quad with ch1,2,-2,-1
        pass
    elif channels == 8: # 7.1
        pass
    else:
        raise ValueError("audio_chunk must have 2 (stereo) or 4 channels")
    

def process_predict_quadro_SED(audio_tensor, sample_rate, skip_model=False, amount_chunks=1):
    """Process quadro (4-channel) audio and return combined events."""
    fl_channel = audio_tensor[:, 0]
    fr_channel = audio_tensor[:, 1]
    rl_channel = audio_tensor[:, 2]
    rr_channel = audio_tensor[:, 3]

    if skip_model:
        fl_events = [{"event": "SKIPEVENTFL", "confidence": 0.0, "orientation": "FL"}]
        fr_events = [{"event": "SKIPEVENTFR", "confidence": 0.0, "orientation": "FR"}]
        rl_events = [{"event": "SKIPEVENTRL", "confidence": 0.0, "orientation": "RL"}]
        rr_events = [{"event": "SKIPEVENTRR", "confidence": 0.0, "orientation": "RR"}]
    else:
        fl_events = predict_event_seperate_SED(fl_channel, sample_rate, orientation="FL",amount_chunks=amount_chunks)
        fr_events = predict_event_seperate_SED(fr_channel, sample_rate, orientation="FR",amount_chunks=amount_chunks)
        rl_events = predict_event_seperate_SED(rl_channel, sample_rate, orientation="RL",amount_chunks=amount_chunks)
        rr_events = predict_event_seperate_SED(rr_channel, sample_rate, orientation="RR",amount_chunks=amount_chunks)
    # Combine events from all channels

    debug = True
    if debug:
        print("FL Events:", fl_events)
        print("FR Events:", fr_events)
        print("RL Events:", rl_events)
        print("RR Events:", rr_events)
    return combine_quadro(fl_events, fr_events, rl_events, rr_events, fl_channel, fr_channel, rl_channel, rr_channel)


    
def predict_event_seperate_SED(audio_arr, sample_rate_device, orientation, cpu=True, amount_chunks=1):
    """
    Predict events using decode_strong for temporal stability.
    
    Uses median filtering and decodes contiguous regions of activity,
    then filters to only return events occurring in the final chunk.
    """
    device = torch.device('cpu') if cpu else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model.eval()
    model.to(device)

    sample_rate = 16_000  # all the models are trained on 16 kHz audio
    segment_duration = 10  # all models are trained on 10-second pieces
    segment_samples = segment_duration * sample_rate

    if len(audio_arr.shape) == 2:
        audio_arr = audio_arr.squeeze(-1)
    
    waveform = librosa.resample(audio_arr, orig_sr=sample_rate_device, target_sr=sample_rate)
    waveform = torch.from_numpy(waveform[None, :]).to(device)
    waveform_len = waveform.shape[1]
    
    total_audio_len = waveform_len / sample_rate  # total duration in seconds
    chunk_duration = total_audio_len / amount_chunks  # duration of each chunk
    
    print(f"Total audio length: {total_audio_len:.2f}s, Chunks: {amount_chunks}, Per chunk: {chunk_duration:.2f}s")
    
    # Encoder for decoding - use original labels for model output
    encoder = ManyHotEncoder(audioset_classes.as_strong_train_classes, audio_len=total_audio_len)

    # Pad to segment length if needed
    waveform_chunk = waveform
    if waveform_chunk.shape[1] < segment_samples:
        pad_size = segment_samples - waveform_chunk.shape[1]
        waveform_chunk = torch.nn.functional.pad(waveform_chunk, (0, pad_size))

    # Run inference
    with torch.no_grad():
        mel = model.mel_forward(waveform_chunk)
        y_strong, _ = model(mel)
    
    # Apply sigmoid to get probabilities
    y_strong = torch.sigmoid(y_strong)
    
    # c_scores shape: [classes, T/numframes]
    c_scores = y_strong.float().squeeze(0)
    
    # Calculate true embedding length based on actual audio (not padded)
    # Each frame is 40ms (10s / 250 frames = 0.04s per frame)
    frame_duration = 0.04
    true_emb_len = int(total_audio_len / frame_duration)
    c_scores = c_scores[:, 0:true_emb_len]
    
    # Transpose to [time_frames, classes] for processing
    c_scores = c_scores.transpose(0, 1).detach().cpu().numpy()
    
    # Apply median filter for temporal stability
    # Ensure window size doesn't exceed number of frames and is odd
    median_window = config["median_window"]
    num_frames = c_scores.shape[0]
    if median_window is not None and num_frames > 1:
        effective_window = min(median_window, num_frames)
        if effective_window % 2 == 0:
            effective_window = max(1, effective_window - 1)
        if effective_window >= 3: 
            c_scores = scipy.ndimage.median_filter(c_scores, (effective_window, 1))
    

    final_chunk_start_time = total_audio_len - chunk_duration
    
    # Apply threshold to get binary predictions
    threshold = config["threshold"]
    pred_binary = c_scores > threshold
    
    decoded_events = encoder.decode_strong(pred_binary) # is temporal stable instead of my argmax
    
    # Filter events to only those that overlap with the final chunk

    final_chunk_events = []
    for event_label, onset, offset in decoded_events:
        if offset > final_chunk_start_time:
            # Event is active in the final chunk
            # Calculate confidence as max score in the overlapping region
            class_idx = encoder.labels.index(event_label)
            
            # Get frame range for the overlapping portion
            overlap_start = max(onset, final_chunk_start_time)
            overlap_start_frame = int(overlap_start / frame_duration)
            overlap_end_frame = min(int(offset / frame_duration), true_emb_len)
            
            if overlap_start_frame < overlap_end_frame:
                confidence = float(c_scores[overlap_start_frame:overlap_end_frame, class_idx].max())
                
                # Map to simplified category
                mapped_event = audioset_classes.as_strong_train_classes[class_idx]
                
                final_chunk_events.append({
                    "event": mapped_event,
                    "original_event": event_label,
                    "confidence": confidence,
                    "orientation": orientation,
                    "onset": onset,
                    "offset": offset
                })
    
    if not final_chunk_events:
        return [{"event": "silence", "confidence": 0.0, "orientation": orientation}]
    
    # Sort by confidence and return top events
    final_chunk_events.sort(key=lambda x: x["confidence"], reverse=True)
    return [final_chunk_events[0]]


