import sounddevice as sd
import numpy as np
from vosk import Model, KaldiRecognizer
import json
import queue
import threading
import sys
import os
from datetime import datetime

def list_audio_devices():
    """List all available audio input devices"""
    print("\nAvailable audio input devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:  # Only show input devices
            print(f"{i}: {device['name']}")
    return devices

# Audio stream parameters
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 2000  # Smaller chunk size for more frequent updates

# Check if model exists
model_path = "models/vosk-model-en-us-0.22"
if not os.path.exists(model_path):
    print("Vosk model not found. Please run download_model.py first to download the model.")
    sys.exit(1)

# Initialize Vosk model
try:
    model = Model(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    sys.exit(1)

# Initialize recognizer with show_partial=True for interim results
rec = KaldiRecognizer(model, SAMPLE_RATE)
rec.SetWords(True)  # Enable word timing information
rec.SetPartialWords(True)  # Enable partial words in interim results

# Create a queue to store audio chunks
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    """This is called for each audio block"""
    if status:
        print(status)
    # Convert the audio data to the right format and add to queue
    audio_data = np.frombuffer(indata, dtype=np.int16)
    # Ensure the audio is mono
    if CHANNELS == 1 and len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    audio_queue.put(audio_data.tobytes())

def process_audio():
    """Process audio chunks from the queue"""
    while True:
        try:
            audio_chunk = audio_queue.get()
            if rec.AcceptWaveform(audio_chunk):
                # Full result
                result = json.loads(rec.Result())
                if result.get("text", "").strip():
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"\n[{timestamp}] {result['text']}")
            else:
                # Partial result
                result = json.loads(rec.PartialResult())
                if result.get("partial", "").strip():
                    print(f"\rPartial: {result['partial']}", end='', flush=True)
        except Exception as e:
            print(f"\nError processing audio: {str(e)}")
            break

def main():
    # List available devices and let user choose
    devices = list_audio_devices()
    try:
        device_id = int(input("\nEnter the number of the input device to use (use Stereo Mix for system audio): "))
        device_info = devices[device_id]
        if device_info['max_input_channels'] == 0:
            print("Selected device has no input channels!")
            return
    except (ValueError, IndexError):
        print("Invalid device selection!")
        return

    try:
        # Start the processing thread
        processing_thread = threading.Thread(target=process_audio, daemon=True)
        processing_thread.start()

        # Start the audio stream with the selected device
        with sd.RawInputStream(samplerate=SAMPLE_RATE,
                             blocksize=CHUNK_SIZE,
                             device=device_id,
                             dtype=np.int16,
                             channels=CHANNELS,
                             callback=audio_callback):
            print("\nStarted listening... Press Ctrl+C to stop")
            print("Transcription will appear below with timestamps...")
            processing_thread.join()

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 