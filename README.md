# Guidar

A transparent overlay designed for hard of hearing individuals who still want to enjoy modern FPS gaming. It pinpoints the direction sounds are coming from, which is crucial in most competitive shooters.

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
6. Run the overlay:
   ```
   npm start
   ```

This will automatically start the Python backend and launch the overlay.

## Example

(screenshot placeholder)

## Bonus: 7.1 Surround via Virtual Audio Cable

The overlay's directional accuracy depends on how many audio channels your system provides. For maximum spatial information, you can spoof a 7.1 surround setup using Virtual Audio Cable (VAC):

1. Install Virtual Audio Cable
2. Open the VAC control panel and configure it with your preferred settings, then click "Set" followed by "Reset"
3. Open Windows Sound Settings
4. Right-click on the VAC output device and select "Configure Speakers"
5. Choose "7.1 Surround" from the list
6. Restart the overlay to use the new channel configuration

## Configuration

Edit `data/config.json` to change the output format:
- `"format": "full"` runs full YAMNet inference (slower, classifies sounds)
- `"format": "cues"` skips classification and only outputs directional cues (faster)

## Troubleshooting

**No audio detected:** Make sure audio is actually playing through your system. The overlay captures loopback audio, so it needs active sound output.

**WebSocket connection failed:** Ensure the Python backend is running before the Electron overlay tries to connect. The `npm start` script includes a delay, but you may need to increase it on slower systems.

**Wrong audio device:** The backend auto-selects the best WASAPI loopback device. If you have multiple audio outputs, make sure your game audio is routed to the one being captured.

## License

YAMNet model by Google Research. See their repository for model licensing details.