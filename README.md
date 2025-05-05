# Speech2RAG: Real-Time Transcription + RAG

This project combines real-time speech recognition with Retrieval-Augmented Generation (RAG) to answer questions about transcribed content on the fly.

## Two Implementations

This project provides two different implementations:

### 1. Original Ollama-based Implementation
- Files: `speech_rag.py` and `streamilt_app.py`
- Uses Ollama for local LLM inference
- Limited model selection to what's available in Ollama
- Requires Ollama to be running separately

### 2. Hugging Face + vLLM Implementation
- Files: `speech_rag_hf.py` and `streamilt_app_hf.py`
- Uses Hugging Face Transformers and vLLM for inference
- Wider range of models available directly from Hugging Face Hub
- Better performance with vLLM acceleration
- More configuration options for model parameters

## Key Differences

| Feature | Ollama Version | HF + vLLM Version |
|---------|----------------|-------------------|
| Model Availability | Limited to Ollama models | Any HF model |
| Inference Speed | Standard | Faster with vLLM |
| GPU Utilization | Via Ollama | Direct PyTorch/vLLM control |
| Configuration | Basic | Advanced options |
| Dependencies | Needs Ollama installed | Self-contained |
| Memory Usage | Managed by Ollama | Directly configurable |

## Installation

### For Ollama Version
```bash
pip install -r requirements.txt
# Also install Ollama separately
```

### For Hugging Face + vLLM Version
```bash
pip install -r requirements_hf.txt
```

## Running the Applications

### Ollama Version
```bash
# Make sure Ollama is running
streamlit run streamilt_app.py
```

### Hugging Face + vLLM Version
```bash
streamlit run streamilt_app_hf.py
```

## Requirements

- VOSK speech recognition models must be downloaded to the `models` directory
- Python 3.8+
- GPU recommended for better performance (especially for HF + vLLM version)

## How It Works

1. Speech is captured in real-time through your microphone
2. VOSK processes the audio and generates a transcript
3. When you ask a question, the system:
   - Chunks the transcript into manageable segments
   - Finds the most relevant chunks using semantic similarity
   - Sends the context and question to the LLM (either via Ollama or directly with HF/vLLM)
   - Returns the answer

## Features

- Real-time speech recognition from microphone or system audio
- Continuous transcription with timestamps
- Ask questions about the transcribed content at any time
- Search through transcription history to find relevant information
- Continues transcribing while answering questions

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Download the speech recognition model:
   ```
   python download_model.py
   ```

3. Run the Speech2RAG system:
   ```
   python speech_rag.py
   ```

## Usage

1. When you run the system, it will display a list of available audio input devices.
2. Select the appropriate device number (use "Stereo Mix" for system audio if available).
3. The system will start transcribing audio in real-time.
4. Type a question at any time to query what has been transcribed so far.
5. The system will search through the transcription history and return the most relevant portions.
6. Type 'exit' or press Ctrl+C to stop the system.

## Notes

- For best results, use a clear audio source and minimize background noise.
- The transcription accuracy depends on the quality of the audio input.
- The RAG system works best with longer transcriptions that contain more context.
- To capture system audio, you may need to enable the "Stereo Mix" device in your sound settings.

## Requirements

- Python 3.7+
- VOSK speech recognition model
- SentenceTransformers for text embeddings
- Sound device for audio capture 