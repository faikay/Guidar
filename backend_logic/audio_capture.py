import numpy as np
import pyaudiowpatch as pyaudio
from backend_logic.utils import _make_callback, find_best_wasapi_loopback_device_pyaudio
import queue
import threading

# TODO let user choose device from a list, and test it


print("\n" * 5)

def get_audio_stream_pyaudiowpatch(chunk_duration=0.96, sample_rate=16000, device=None):
    """Get audio chunks using PyAudioWPatch (callback-based WASAPI loopback).

    This implementation uses a PyAudio `stream_callback` to collect frames into
    a small `queue.Queue` and yields numpy arrays from the generator. Using the
    callback avoids blocking reads and is better suited to WASAPI loopback on
    Windows.

    Yields tuples `(arr, sample_rate)` where `arr` is a float32 numpy array
    shaped `(samples,)` or `(samples, channels)` depending on the device.
    """

    pa = pyaudio.PyAudio()

    if device is None:
        device,_ = find_best_wasapi_loopback_device_pyaudio()
        if device is None:
            raise RuntimeError("No wasapi based PyAudio loopback device found")

    dev_info = pa.get_device_info_by_index(device)

    # Use device's default sample rate if available, else use provided sample_rate
    sample_rate = int(dev_info.get('defaultSampleRate', sample_rate))
    channels = max(dev_info.get('maxInputChannels'), 2) # prior code should ensure stereo
    chunk_samples = int(chunk_duration * sample_rate)
    dtype_candidates = [pyaudio.paFloat32, pyaudio.paInt16]  # prefer float32-> first

    # background queue to ensure we do not mis audio data
    data_q = queue.Queue(maxsize=15)
    stop_event = threading.Event()

    chosen_dtype = None
    stream = None

    print(f"Starting audio stream with sample_rate={sample_rate}, channels={channels}, chunk_samples={chunk_samples}")
    for dtype in dtype_candidates:
        try:
            callback_fn = _make_callback(data_q)
            stream = pa.open(format=dtype,
                                channels=channels,
                                rate=sample_rate,
                                input=True,
                                frames_per_buffer=chunk_samples,
                                input_device_index=device,
                                stream_callback=callback_fn)
            chosen_dtype = dtype
            break
        except Exception as e:
            stream = None
    if stream is None:
        raise RuntimeError("Failed to open PyAudio input stream with available formats")

    # Start the stream (callback will begin filling the queue)
    stream.start_stream()

    try:
        while not stop_event.is_set(): # for final shutdown, not queue 
            try:
                raw = data_q.get(timeout=1)
            except Exception:
                # Timeout — check stream status and continue
                if not stream.is_active():
                    break
                continue

            if chosen_dtype == pyaudio.paFloat32:
                 # frombuffer on an immutable `bytes` returns a read-only view;
                 # make an explicit copy so downstream code (torch.from_numpy)
                 # can safely assume a writable, contiguous array.
                 arr = np.frombuffer(raw, dtype=np.float32).copy()
            else:
                 # Convert int16 -> float32, and copy to ensure writable/contiguous
                 arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32).copy() / 32768.0

            # If channels > 1, ensure proper shape
            if channels > 1:
                try:
                    arr = arr.reshape(-1, channels)
                except Exception:
                    raise Exception("channel reshape fail (divisibibility)") # todo: fix
            yield arr, sample_rate

    finally:
        try:
            stop_event.set()
            if stream is not None:
                try:
                    stream.stop_stream()
                except Exception:
                    pass
                try:
                    stream.close()
                except Exception:
                    pass
        finally:
            try:
                pa.terminate()
            except Exception:
                pass
    
