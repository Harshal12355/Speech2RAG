import sounddevice as sd
import numpy as np
from vosk import Model, KaldiRecognizer
import json
import queue
import threading
import sys
import os
import time
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Audio stream parameters
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 2000  # Smaller chunk size for more frequent updates

class Speech2RAG:
    def __init__(self, model_path="models/vosk-model-small-en-us-0.22"):
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"Vosk model not found at {model_path}. Please run download_model.py first.")
            sys.exit(1)
            
        # Initialize Vosk model
        try:
            self.model = Model(model_path)
            print("Speech recognition model loaded successfully!")
        except Exception as e:
            print(f"Error loading speech model: {str(e)}")
            sys.exit(1)
            
        # Initialize embeddings model for RAG
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Embedding model loaded successfully!")
        except Exception as e:
            print(f"Error loading embedding model: {str(e)}")
            sys.exit(1)
            
        # Initialize recognizer with show_partial=True for interim results
        self.rec = KaldiRecognizer(self.model, SAMPLE_RATE)
        self.rec.SetWords(True)  # Enable word timing information
        self.rec.SetPartialWords(True)  # Enable partial words in interim results
        
        # Create a queue to store audio chunks
        self.audio_queue = queue.Queue()
        
        # Store transcription history
        self.transcription_history = []
        self.partial_transcription = ""
        
        # Threading control
        self.is_running = False
        self.audio_thread = None
        self.processing_thread = None
        
    def list_audio_devices(self):
        """List all available audio input devices"""
        print("\nAvailable audio input devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:  # Only show input devices
                print(f"{i}: {device['name']}")
        return devices
         
    def audio_callback(self, indata, frames, time, status):
        """This is called for each audio block"""
        if status:
            print(status)
        # Convert the audio data to the right format and add to queue
        audio_data = np.frombuffer(indata, dtype=np.int16)
        # Ensure the audio is mono
        if CHANNELS == 1 and len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        self.audio_queue.put(audio_data.tobytes())
    
    def process_audio(self):
        """Process audio chunks from the queue"""
        while self.is_running:
            try:
                audio_chunk = self.audio_queue.get(timeout=1)
                if self.rec.AcceptWaveform(audio_chunk):
                    # Full result
                    result = json.loads(self.rec.Result())
                    if result.get("text", "").strip():
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        transcribed_text = result['text']
                        print(f"\n[{timestamp}] {transcribed_text}")
                        
                        # Add to history with timestamp
                        self.transcription_history.append({
                            "timestamp": timestamp,
                            "text": transcribed_text
                        })
                else:
                    # Partial result
                    result = json.loads(self.rec.PartialResult())
                    partial = result.get("partial", "").strip()
                    if partial:
                        self.partial_transcription = partial
                        print(f"\rPartial: {partial}", end='', flush=True)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"\nError processing audio: {str(e)}")
                if not self.is_running:
                    break
    
    def get_full_transcript(self):
        """Get the complete transcript so far"""
        full_text = " ".join([entry["text"] for entry in self.transcription_history])
        if self.partial_transcription:
            full_text += " " + self.partial_transcription
        return full_text
    
    def chunk_transcript(self, text, chunk_size=150, overlap=50):
        """Split transcript into overlapping chunks for better retrieval"""
        words = text.split()
        if len(words) <= chunk_size:
            return [text]
            
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks
    
    def answer_question(self, question, top_k=3):
        """Answer a question based on the current transcript"""
        # Get full transcript
        transcript = self.get_full_transcript()
        if not transcript.strip():
            return "No transcription available yet."
            
        # Chunk the transcript
        chunks = self.chunk_transcript(transcript)
        
        # Encode question and chunks
        question_embedding = self.embedding_model.encode([question])[0]
        chunk_embeddings = self.embedding_model.encode(chunks)
        
        # Calculate similarities
        similarities = cosine_similarity([question_embedding], chunk_embeddings)[0]
        
        # Get top k most relevant chunks
        top_indices = similarities.argsort()[-top_k:][::-1]
        relevant_chunks = [chunks[i] for i in top_indices]
        
        # Format response with relevant transcript portions
        response = "Based on the transcription, here are the most relevant portions:\n\n"
        for i, chunk in enumerate(relevant_chunks):
            response += f"Relevant part {i+1}:\n{chunk}\n\n"
        
        response += "Note: The transcription is ongoing and more information may become available."
        return response
    
    def start(self):
        """Start the transcription and RAG system"""
        # List available devices and let user choose
        devices = self.list_audio_devices()
        try:
            device_id = int(input("\nEnter the number of the input device to use (use Stereo Mix for system audio): "))
            device_info = devices[device_id]
            if device_info['max_input_channels'] == 0:
                print("Selected device has no input channels!")
                return
        except (ValueError, IndexError):
            print("Invalid device selection!")
            return
            
        self.is_running = True
        
        # Start the processing thread
        self.processing_thread = threading.Thread(target=self.process_audio, daemon=True)
        self.processing_thread.start()
        
        # Start the audio stream with the selected device
        self.audio_thread = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SIZE,
            device=device_id,
            dtype=np.int16,
            channels=CHANNELS,
            callback=self.audio_callback
        )
        self.audio_thread.start()
        
        print("\nStarted listening... Transcription will appear below with timestamps...")
        print("Type a question at any time to query what has been transcribed so far.")
        print("Type 'exit' or press Ctrl+C to stop.")
        
        # Command loop
        try:
            while self.is_running:
                user_input = input("\nQuestion (or 'exit'): ")
                if user_input.lower() == 'exit':
                    break
                    
                if user_input.strip():
                    print("\nSearching transcription...")
                    answer = self.answer_question(user_input)
                    print("\n" + answer + "\n")
                
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
    
    def stop(self):
        """Stop the transcription and RAG system"""
        print("\nStopping transcription and RAG system...")
        self.is_running = False
        if self.audio_thread:
            self.audio_thread.stop()
            self.audio_thread.close()
        print("Stopped!")

if __name__ == "__main__":
    rag_system = Speech2RAG()
    rag_system.start() 