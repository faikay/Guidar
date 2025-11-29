import asyncio
from backend_logic.utils import find_best_wasapi_loopback_device_pyaudio, peek_queue
import websockets
import json
import os
from backend_logic.audio_capture import get_audio_stream_pyaudiowpatch
#from backend_logic.model_inference import predict_events
from PretrainedSED.SED_inference_custom import predict_events_SED as predict_events
import traceback

# A set of currently connected clients (websocket objects)
CONNECTED_CLIENTS = set()

# queue for events detected by the model; broadcasted to clients by a separate
# task. This is created when the server starts.
EVENT_QUEUE = None
AUDIO_QUEUE = None


with open(os.path.join(os.path.dirname(__file__), '..', "data" , 'config.json'), 'r') as f:
    config = json.load(f)

SETUP = config.get("format", "cues")
min_queue_size = 3 #config.get("min_queue_size", 3)
skip_model = True if SETUP == "cues" else False

async def handler(websocket):
    """Per-client handler: register client and wait for disconnect.

    The broadcasting of detected events to all clients is handled by the
    `event_broadcaster` task, which runs once when the server starts.
    """
    # Register client
    CONNECTED_CLIENTS.add(websocket)
    try:
        await websocket.send(json.dumps({"event": "Connected!!", "confidence": 0.0, "orientation": "NAN"}))
        await websocket.wait_closed()

    finally:
        CONNECTED_CLIENTS.discard(websocket)

async def start_websocket_server():
    global EVENT_QUEUE, AUDIO_QUEUE
    EVENT_QUEUE = asyncio.Queue(maxsize=15)
    AUDIO_QUEUE = asyncio.Queue(maxsize=15)
    queue_min_condition = asyncio.Condition()

    
    async def wait_for_audio_q_size(condition,audio_queue,n=3):
        async with condition:
            await condition.wait_for(lambda: audio_queue.qsize() >= n)

    async def audio_producer(audio_q: asyncio.Queue):
        """Produce events from audio and put them on the queue."""
        device,channels = find_best_wasapi_loopback_device_pyaudio()
        generator = get_audio_stream_pyaudiowpatch(device=device)
        while True:
            try:
                audio_chunk, sample_rate = await asyncio.to_thread(next, generator)

                async with queue_min_condition:
                    await audio_q.put((audio_chunk, sample_rate,channels))
                    queue_min_condition.notify_all()

                    #print("[audio_producer] Got audio chunk")
            except StopIteration:
                #print("[audio_producer] Audio generator no longer receiving data")
                #break       it should not stop in normal operation, 
                continue
            except Exception as e:
                async with queue_min_condition:
                    await audio_q.put({"error": f"Audio capture error: {str(e)}"})
                    queue_min_condition.notify_all()
                    await asyncio.sleep(0.1)
                continue
    
    async def event_producer(audio_q: asyncio.Queue, event_q: asyncio.Queue):
        while True:
            try:
                await wait_for_audio_q_size(queue_min_condition,audio_q,min_queue_size)
                
                async with queue_min_condition:
                    audio_chunks, sample_rates, channels_audios = peek_queue(audio_q,min_queue_size) # get min_queue_size items, remove only oldest 
                    _ = await audio_q.get()

                events = await asyncio.to_thread(predict_events, audio_chunks, sample_rates, channels_audios, skip_model=skip_model)
                print(f"[audio_producer] Inference complete, events: {events}")
            except Exception as e:
                print(f"[audio_producer] Inference error: {e}")
                traceback.print_exc()
                await event_q.put({"error": f"Inference error: {str(e)}"})
                continue
               
            
            #print(f"[audio_producer] Detected events: {q.qsize()}")
            await event_q.put(events)


    async def event_broadcaster(q: asyncio.Queue):
        """Read events from the queue and broadcast to all connected clients."""
        try:
            while True:
                #print("[event_broadcaster] Waiting for events...")
                events = await q.get()
                #print(f"[event_broadcaster] Got events: {events}")

                if CONNECTED_CLIENTS:
                    #print(f"[event_broadcaster] Broadcasting to {len(CONNECTED_CLIENTS)} clients")
                    data = json.dumps(events)
                    coros = []
                    coros = []
                    for ws in list(CONNECTED_CLIENTS):
                        coros.append(ws.send(data))

                    if coros:
                        results = await asyncio.gather(*coros, return_exceptions=True)
                        # If at least one send succeeded (not exception) we consider
                        # the event delivered and will not re-queue it. If all
                        # sends failed, requeue for a retry later.
                        successes = sum(1 for r in results if not isinstance(r, Exception))
                        if successes == 0:
                            print("[event_broadcaster] All sends failed — requeueing event")
                            await asyncio.sleep(0.2)
                            await q.put(events)
                else:
                    print("[event_broadcaster] No connected clients to broadcast to.")
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[event_broadcaster] Exception: {e}\n{tb}")
            # Log the exception as an event in the queue for UI/consumer
            await q.put({"error": f"event_broadcaster exception: {e}", "traceback": tb})

    # Start producer and broadcaster tasks while server runs
    audio_producer_task = asyncio.create_task(audio_producer(AUDIO_QUEUE))
    event_producer_task = asyncio.create_task(event_producer(AUDIO_QUEUE, EVENT_QUEUE))
    broadcaster_task = asyncio.create_task(event_broadcaster(EVENT_QUEUE))

    try:
        async with websockets.serve(handler, "127.0.0.1", 8080):
            print("WebSocket server started at ws://127.0.0.1:8080")
            await asyncio.Future()  # Run forever
    finally:
        audio_producer_task.cancel()
        broadcaster_task.cancel()
        event_producer_task.cancel()