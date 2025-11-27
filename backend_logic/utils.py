import json
import os
import pyaudiowpatch as pyaudio
import numpy as np
import torch
import yaml

# ---- general utility functions ----

# Convert torch tensors to numpy on CPU
def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

def softmax_with_temp(vals, temp=1.0):
    vals = np.array(vals, dtype=np.float64)
    if temp == 0:
        one_hot = np.zeros_like(vals)
        one_hot[np.argmax(vals)] = 1.0
        return one_hot
    shifted = vals / temp
    shifted -= shifted.max()                 # numerical stability
    exps = np.exp(shifted)
    s = exps.sum()
    if s == 0:
        return np.ones_like(vals) / len(vals)
    return exps / s


def get_indices_from_dict(element, lst):
    lst_events = [d["event"] for d in lst]
    return [i for i, x in enumerate(lst_events) if x == element]

def get_yamnet_classes():

    with open(os.path.join(os.path.dirname(__file__), "..", "data" , 'config.json'), 'r') as f:
        config = json.load(f)
        config_format = config.get("format", "cues")
        print(f"Config format: {config_format}")
    if config_format == "cues":
        cat_meta_file = os.path.join(os.path.dirname(__file__),"..","data", "other_and_silence.yaml") # other_and_silence , yamnet_custom_meta
    else:
        cat_meta_file = os.path.join(os.path.dirname(__file__),"..","data", "yamnet_custom_meta.yaml") # other_and_silence , yamnet_custom_meta
    with open(cat_meta_file) as f:
        cat_meta = yaml.safe_load(f) 
    return [ item["custom_name"] for item in cat_meta ]  # only take the name
    
def _energy(channel_data):
    return np.sum(np.square(channel_data))


def within_margin(values, margin=0.20):
    """
    Return True if all values differ by at most `margin` (e.g. 0.10 for 10%)
    relative to the maximum value in the list.
    """
    if not values:
        return True  # or False, depending on how you want to treat empty lists

    v_min = min(values)
    v_max = max(values)

    # If everything is exactly the same (including all zeros)
    if v_max == 0:
        return v_min == 0

    # Relative spread between min and max
    rel_diff = (v_max - v_min) / v_max

    return rel_diff <= margin



# ---- audio read and bbroadasting utility fucs ------


def dummy_predict_events(audio_chunk, sample_rate):
    # dummy  
    return [{"event": "dummy", "confidence": 0.0, "orientation": "NAN"}]

def find_first_wasapi_loopback_device_pyaudio():
    """Find a WASAPI loopback device index using PyAudioWPatch. Returns index or None."""

    pa = pyaudio.PyAudio()
    try:
        # PyAudioWPatch adds loopback devices — prefer explicit 'loopback' in name
        for i in range(pa.get_device_count()):
            try:
                info = pa.get_device_info_by_index(i)
            except Exception:
                continue
            name = info.get('name', '').lower()
            isloop = bool(info.get('isLoopbackDevice', ''))

            if 'wasapi' in name or 'loopback' in name or isloop:
                channels = info.get('maxInputChannels', 0)
                if channels >= 2: #  stereo loopback
                    print(f"Selected PyAudio loopback device {i}: {info.get('name')}, with {channels} input channels")
                    return i, channels
            
                raise RuntimeError("No wasapi based PyAudio loopback device with stereo found")
    finally:
        pa.terminate()
    return None


# Find the WASAPI/loopback device with the highest input channel count
def find_best_wasapi_loopback_device_pyaudio():
    """Find the WASAPI loopback device index with the highest input channel count using PyAudioWPatch. Returns index or None."""
    pa = pyaudio.PyAudio()
    best_index = None
    best_channels = -1
    try:
        for i in range(pa.get_device_count()):
            try:
                info = pa.get_device_info_by_index(i)
            except Exception:
                continue
            name = info.get('name', '').lower()
            isloop = bool(info.get('isLoopbackDevice', ''))
            if 'wasapi' in name or 'loopback' in name or isloop:
                channels = info.get('maxInputChannels', 0)
                if channels > best_channels:
                    best_channels = channels
                    best_index = i
        if best_index is not None:
            info = pa.get_device_info_by_index(best_index)
            print(f"Best PyAudio WASAPI loopback device {best_index}: {info.get('name')}, with {info.get('maxInputChannels')} input channels")
            return best_index, best_channels
    finally:
        pa.terminate()
    return None




def _make_callback(q):
    # callback closure: gets raw bytes and pushes to queue
    def callback(in_data, frame_count, time_info, status):
        try:
            # Put raw bytes into the queue; if full, drop oldest
            if q.full():
                try:
                    _ = q.get_nowait()
                except Exception:
                    pass
            q.put_nowait(in_data)
        except Exception:
            # If anything goes wrong, just continue
            pass
        return (None, pyaudio.paContinue)
    return callback


# ---- combine utility functions ----


# TODO TD based logic to determine orientation more robustly -> alledgely not the case for playback/loopback, need mics

def quadro_quadrant_from_azimuth(az_deg):
    """
    Given azimuth (0–360°), return which surround quadrant it belongs to:
    'FL', 'FR', 'RR', 'RL'
    """

    az = az_deg % 360.0  # safety

    # FL: from 315°→360° OR 0°→45°
    if az >= 315 or az < 45:
        return "FL",0

    # FR: 45°→135°
    if 45 <= az < 135:
        return "FR",1   
    # RR: 135°→225°
    if 135 <= az < 225:
        return "RR",3

    # RL: 225°→315°
    if 225 <= az < 315:
        return "RL",2
    # should never happen
    return "UNKNOWN"


def azimuth_from_three_energy(front, back, left, right):
    y = front - back
    x = right - left

    theta = np.arctan2(y, x)
    theta_deg = np.degrees(theta)

    if theta_deg < 0:
        theta_deg += 360.0
    
    theta_deg = theta_deg +90.0
    theta_deg = float(theta_deg % 360.0)
    
    #_, quadrant_idx = quadrant_from_azimuth(theta_deg)
    
    return theta_deg

def seven_one_quadrant_from_azimuth(az_deg):
    """
    Given azimuth (0–360°), return which 7.1 sector it belongs to.
    Returns (name, index) where index matches combine_7one order:
    0: FL, 1: FR, 2: RL, 3: RR, 4: SL, 5: SR
    """
    az = az_deg % 360.0

    # FR (30°): 0° -> 60°
    if 0 <= az < 60:
        return "FR", 1
    # SR (90°): 60° -> 120°
    if 60 <= az < 120:
        return "SR", 5
    # RR (150°): 120° -> 180°
    if 120 <= az < 180:
        return "RR", 3
    # RL (210°): 180° -> 240°
    if 180 <= az < 240:
        return "RL", 2
    # SL (270°): 240° -> 300°
    if 240 <= az < 300:
        return "SL", 4
    # FL (330°): 300° -> 360°
    return "FL", 0
