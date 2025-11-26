import asyncio
from backend_logic.audio_capture import get_audio_stream_pyaudiowpatch
from backend_logic.utils import find_first_wasapi_loopback_device_pyaudio
from backend_logic.websocket_server import start_websocket_server


if __name__ == "__main__":
    test_wasapi_loopback = False 


    if test_wasapi_loopback:
        DEVICE_INDEX = find_first_wasapi_loopback_device_pyaudio()
        generator = get_audio_stream_pyaudiowpatch(device=DEVICE_INDEX)
        for _ in range(5):
            audio_chunk, sample_rate = next(generator)
            print(f"Got audio chunk with shape {audio_chunk.shape} at sample rate {sample_rate}")
    else:
        asyncio.run(start_websocket_server())