# Speech2RAG

A real-time speech recognition system with Retrieval Augmented Generation (RAG) capabilities. This system transcribes audio in real-time and allows you to ask questions about the transcribed content while the transcription continues.

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