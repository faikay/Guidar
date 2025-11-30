# Guidar

A transparent overlay designed for hard of hearing individuals who still want to enjoy modern FPS gaming. It pinpoints the direction sounds are coming from, which is crucial in most competitive shooters.


## Example

https://github.com/user-attachments/assets/f4da2671-563f-4bcd-925b-5a5ed9ff5fd6

## How It Works

**Backend:** An async Python WebSocket server streams system audio in chunks using the Windows WASAPI loopback protocol. Audio data is processed through Google's YAMNet model to classify sound sources. YAMNet's 521 classes are simplified into 6 superclasses for gaming relevance: "gun", "footsteps", "voices", "silence", "vehicle", and "general noise". Sound direction is determined by comparing the energy levels across different audio channels, whilst taking into account the detected superclasses. Supports stereo, quadraphonic, 5.1, and 7.1 speaker configurations.

**Frontend:** An Electron-based transparent overlay that displays detected sounds with distinct symbols and colors for each class, making it easy to differentiate between gunfire, footsteps, and other audio cues at a glance.

## Project Structure

```
Guidar/
├── main.py                  # Entry point, starts the WebSocket server
├── environment.yml          # Conda environment specification
├── backend_logic/
│   ├── audio_capture.py     # WASAPI loopback audio streaming
│   ├── model_inference.py   # YAMNet inference and class simplification
│   ├── combine.py           # Channel energy comparison for directionality
│   ├── websocket_server.py  # Async WebSocket server broadcasting events
│   └── utils.py             # Device detection and helper functions
├── data/
│   ├── config.json          # Runtime configuration (output format)
│   └── svg/                  # Icon assets for each sound class
├── electron_overlay/
│   ├── main.js              # Electron window setup (transparent, always-on-top)
│   ├── index.html           # Overlay UI and WebSocket client
│   └── package.json         # Node dependencies and start scripts
└── ui/
    └── index.html           # Alternative browser-based UI (for testing)
```

## Requirements

- Windows 10/11 (WASAPI is Windows-only)
- Node.js (for Electron frontend)
- Conda (for Python environment management)
- A working audio output device (the overlay captures system audio via loopback)

## Installation

1. Clone the repository
2. Make sure Node.js and Conda are installed on your system
3. Create the conda environment:
   ```
   conda env create -f environment.yml
   ```
4. Activate the environment:
   ```
   conda activate audioenv
   ```
5. Install the Electron dependencies:
   ```
   cd electron_overlay
   npm install
   ```
6. Run the overlay in a Node.js CMD (not Powershell):
   ```
   npm start
   ```

This will automatically start the Python backend and launch the overlay.

## Bonus: 7.1 Surround via Virtual Audio Cable

The overlay's directional accuracy depends on how many audio channels your system provides. For maximum spatial information, you can spoof a 7.1 surround setup using Virtual Audio Cable (VAC):

1. Install [Virtual Audio Cable](https://vac.muzychenko.net/en/download.htm)
2. Open the VAC control panel by runnin it as admin, and configure it with these settings:
<img width="1318" height="315" alt="image" src="https://github.com/user-attachments/assets/a8069aa7-91f8-4f26-bb2f-1352d3e5c605" />
4. Click "Set" followed by "Reset"
5. Click on "Audio Properties"
6. Click on the VAC output device and select "Configure"
7. Choose "7.1 Surround" from the list
8. Restart the overlay to use the new channel configuration

## Configuration
Depending on your setup, one can consider two configurations ("full" should be able to run on most systems):
- `"format": "full"` runs full YAMNet inference (slower, classifies sounds)
- `"format": "cues"` skips classification and only outputs directional cues (faster)
Edit `data/config.json` to change the format.

## Troubleshooting

**No audio detected:** Make sure audio is actually playing through your system. The overlay captures loopback audio, so it needs active sound output.

**WebSocket connection failed:** Ensure the Python backend is running before the Electron overlay tries to connect. The `npm start` script includes a delay, but you may need to increase it on slower systems.

**Wrong audio device:** The backend auto-selects the best WASAPI loopback device. If you have multiple audio outputs, make sure your game audio is routed to the one being captured.

## License 

-YAMNet model by Google Research. See [their repository](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet) for model licensing details. Utilised the pytorch (instead of TF) port from the torch-vggish-yamnet python package 

-[Credits for the original example video](https://www.youtube.com/watch?v=BMqDCws-C8o)
