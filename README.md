
# ComfyUI CSM Node

A ComfyUI node for the specific CSM model `sesame/csm-1b`.
This node uses the `transformers` library for model loading and inference.

## Features

- **Text-to-Speech**: Generate speech from text using the CSM 1B model.
- **Voice Cloning / Context**: Provide reference audio and text to clone a voice or provide context.
- **Automatic Downloading**: Automatically downloads the model from Hugging Face (`sesame/csm-1b`).
- **Caching**: keeps the model loaded in memory (VRAM) for faster subsequent generations, offloading to CPU when a new model is loaded or node is unloaded (handled by python gc/ComfyUI).

## Installation

1.  Navigate to your ComfyUI `custom_nodes` directory.
2.  Clone this repository:
    ```bash
    git clone https://github.com/SesameAILabs/ComfyUI_CSM
    ```
    (Or copy the `ComfyUI_CSM` folder to `custom_nodes`)
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    **Note**: Requires `transformers>=4.49.0`.

## Usage

### Node: **CSM Transformers**

**Inputs:**
- `text`: The text you want the model to speak.
- `speaker_id`: Identifier for the speaker (e.g., "0", "1").
- `model_id`: The Hugging Face model ID. Default: `sesame/csm-1b`.

**Optional Inputs:**
- `ref_audio`: A reference audio clip (Audio input) to use for voice context.
- `ref_text`: The text corresponding to the `ref_audio`.

**Outputs:**
- `audio`: The generated audio waveform (24kHz).

## Requirements

- Python 3.8+
- PyTorch
- `transformers>=4.49.0`
- `torchaudio`
