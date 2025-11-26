import torch
import torch.nn.functional as F
import numpy as np
import librosa
import os
from torch_vggish_yamnet import yamnet
from torch_vggish_yamnet.input_proc import WaveformToInput
import yaml
from backend_logic.utils import get_yamnet_classes
from backend_logic.combine import combine_stereo, combine_quadro, combine_7one


try: # model load
    model = yamnet.yamnet(pretrained=True)
    model.eval()
    waveform_to_input = WaveformToInput()

    class_names = get_yamnet_classes()
except Exception as e:
    print(f"Error loading YAMNet model or category metadata: {e}")
    model = None
    waveform_to_input = None
    class_names = []
    exit()


def predict_events(audio_chunk, sample_rate,channels,skip_model=False):
    if model is None or waveform_to_input is None:
        return [{"event": "model_not_loaded", "confidence": 0.0, "orientation": "NAN"}]
    
    if isinstance(audio_chunk, np.ndarray):
        audio_tensor = torch.from_numpy(np.ascontiguousarray(audio_chunk, dtype=np.float32))
    elif isinstance(audio_chunk, torch.Tensor):
        audio_tensor = audio_chunk.to(dtype=torch.float32)
    else:
        raise TypeError("audio_chunk must be a numpy array or torch Tensor")

    if channels == 2:
        return process_predict_stereo(audio_tensor, sample_rate, skip_model)
    elif channels == 4:
        return process_predict_quadro(audio_tensor, sample_rate, skip_model)
    elif channels == 6: # 5.1 = quad with ch1,2,-2,-1
        audio_tensor = torch.cat([audio_tensor[:, :2], audio_tensor[:, -2:]], dim=1)  
        return process_predict_quadro(audio_tensor, sample_rate, skip_model)
    elif channels == 8: # 7.1
        return process_predict_7one(audio_tensor, sample_rate, skip_model)
    else:
        raise ValueError("audio_chunk must have 2 (stereo) or 4 channels")


# channel independent prediction
def predict_events_separate(audio_tensor, sample_rate,orientation="NAN"):
    """Predict events from an audio chunk using YAMNet.

    Accepts NumPy arrays (as produced by `audio_capture`) or torch tensors.
    Returns a list of detected events with confidence scores (top-1).
    """
    # input shape : [samples,channels] /  [samples,] , needed shape: [channels, samples]
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(-1)
    audio_tensor = audio_tensor.T

    # Preprocess and run inference

    if audio_tensor.shape[1] <500:
        print("Input audio too short or undersampled", audio_tensor.shape)
        return [{"event": "input too short/undersampled", "confidence": 0.0, "orientation": "NAN"}]
    input_tensor = waveform_to_input(audio_tensor, sample_rate) # input tens[C,T] -> [1/N,C,T]
    
    with torch.no_grad():
        _,  scores = model.forward(input_tensor,to_prob=True)

    probs = F.softmax(scores, dim=-1)
    max_prob, max_idx = torch.max(probs, dim=-1)
    confidence = max_prob.item()
    event = class_names[max_idx.item()] if class_names else str(int(max_idx.item()))

    return [{"event": event, "confidence": confidence, "orientation": orientation}]






# -------- stereo -------

def process_predict_stereo(audio_tensor, sample_rate, skip_model=False):
    """Process stereo audio and return combined events."""
    left_channel = audio_tensor[:, 0]
    right_channel = audio_tensor[:, 1]

    if skip_model:
        left_events = [{"event": "SKIPEVENTL", "confidence": 0.0, "orientation": "L"}]
        right_events = [{"event": "SKIPEVENTR", "confidence": 0.0, "orientation": "R"}]
    else:
        left_events = predict_events_separate(left_channel, sample_rate, orientation="L")
        right_events = predict_events_separate(right_channel, sample_rate, orientation="R")

    return combine_stereo(left_events, right_events, left_channel, right_channel)

# -------- quad (and 5.1)  -------

def process_predict_quadro(audio_tensor, sample_rate, skip_model=False):
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
        fl_events = predict_events_separate(fl_channel, sample_rate, orientation="FL")
        fr_events = predict_events_separate(fr_channel, sample_rate, orientation="FR")
        rl_events = predict_events_separate(rl_channel, sample_rate, orientation="RL")
        rr_events = predict_events_separate(rr_channel, sample_rate, orientation="RR")
    # Combine events from all channels
    return combine_quadro(fl_events, fr_events, rl_events, rr_events, fl_channel, fr_channel, rl_channel, rr_channel)

# -------- 7.1 -------

def process_predict_7one(audio_tensor, sample_rate, skip_model=False):
    """
    Process 7.1 (8-channel) audio and return combined events.
    
    7.1 expected WASAPI/FFmpeg order:
        0: FL
        1: FR
        2: FC
        3: LFE
        4: BL
        5: BR
        6: SL
        7: SR

    We ignore FC + LFE (non-directional) and keep six directional speakers:
        FL, FR, BL, BR, SL, SR
    """
    # Extract channel tensors
    fl_channel = audio_tensor[:, 0]   # Front Left
    fr_channel = audio_tensor[:, 1]   # Front Right
    # FC = audio_tensor[:, 2]         # Ignored
    # LFE = audio_tensor[:, 3]        # Ignored
    bl_channel = audio_tensor[:, 4]   # Back Left
    br_channel = audio_tensor[:, 5]   # Back Right
    sl_channel = audio_tensor[:, 6]   # Side Left
    sr_channel = audio_tensor[:, 7]   # Side Right

    if skip_model:
        fl_events = [{"event": "SKIPEVENT_FL", "confidence": 0.0, "orientation": "FL"}]
        fr_events = [{"event": "SKIPEVENT_FR", "confidence": 0.0, "orientation": "FR"}]
        bl_events = [{"event": "SKIPEVENT_BL", "confidence": 0.0, "orientation": "BL"}]
        br_events = [{"event": "SKIPEVENT_BR", "confidence": 0.0, "orientation": "BR"}]
        sl_events = [{"event": "SKIPEVENT_SL", "confidence": 0.0, "orientation": "SL"}]
        sr_events = [{"event": "SKIPEVENT_SR", "confidence": 0.0, "orientation": "SR"}]
    else:
        fl_events = predict_events_separate(fl_channel, sample_rate, orientation="FL")
        fr_events = predict_events_separate(fr_channel, sample_rate, orientation="FR")
        bl_events = predict_events_separate(bl_channel, sample_rate, orientation="BL")
        br_events = predict_events_separate(br_channel, sample_rate, orientation="BR")
        sl_events = predict_events_separate(sl_channel, sample_rate, orientation="SL")
        sr_events = predict_events_separate(sr_channel, sample_rate, orientation="SR")

    # Combine all directional channels into one structure
    return combine_7one(
        fl_events, fr_events, bl_events, br_events, sl_events, sr_events,
        fl_channel, fr_channel, bl_channel, br_channel, sl_channel, sr_channel
    )

   

# gun, foot, vehicle, voice, 
#7.1 = quad + pure sides
# 5.1 = quad 
# TODO increase sampling secs in stream to allow when we downsample to 16kHz we still good data

