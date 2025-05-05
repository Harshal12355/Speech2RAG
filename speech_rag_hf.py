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
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Try to import vLLM, but make it optional
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("vLLM not available. Will use standard HuggingFace pipelines instead.")

# Audio stream parameters
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 2000  # Smaller chunk size for more frequent updates

# Default models
DEFAULT_VOSK_MODEL = "models/vosk-model-small-en-us-0.15"
DEFAULT_HF_MODEL = "google/gemma-1.1-2b-it"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class Speech2RAG_HF:
    def __init__(self, model_path=DEFAULT_VOSK_MODEL, 
                 hf_model_name=DEFAULT_HF_MODEL, 
                 embedding_model_name=DEFAULT_EMBEDDING_MODEL,
                 use_vllm=VLLM_AVAILABLE,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize Speech2RAG with Hugging Face models
        
        Args:
            model_path: Path to VOSK model for speech recognition
            hf_model_name: Hugging Face model name for text generation
            embedding_model_name: Model for text embeddings
            use_vllm: Whether to use vLLM for acceleration
            device: Device to use for inference (cuda/cpu)
        """
        # Check if VOSK model exists
        if not os.path.exists(model_path):
            print(f"Vosk model not found at {model_path}. Please download it first.")
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
            self.embedding_model = SentenceTransformer(embedding_model_name)
            print(f"Embedding model '{embedding_model_name}' loaded successfully!")
        except Exception as e:
            print(f"Error loading embedding model: {str(e)}")
            sys.exit(1)
        
        # Initialize LLM
        self.hf_model_name = hf_model_name
        self.device = device
        
        # Override use_vllm if not available
        if use_vllm and not VLLM_AVAILABLE:
            print("vLLM requested but not available. Falling back to standard HF pipeline.")
            use_vllm = False
        
        self.use_vllm = use_vllm
        
        try:
            if use_vllm:
                # vLLM for faster inference
                self.llm = LLM(model=hf_model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
                print(f"vLLM model '{hf_model_name}' loaded successfully!")
            else:
                # Standard HF pipeline
                self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
                self.llm = AutoModelForCausalLM.from_pretrained(
                    hf_model_name, 
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map=device
                )
                print(f"Hugging Face model '{hf_model_name}' loaded successfully!")
        except Exception as e:
            print(f"Error loading LLM model: {str(e)}")
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
    
    def answer_question_with_hf(self, question, top_k=3, max_new_tokens=512, temperature=0.3):
        """Answer a question using Hugging Face models with RAG"""
        transcript = self.get_full_transcript()
        if not transcript.strip():
            return "No transcription available yet."

        # Chunk the transcript for better retrieval
        chunks = self.chunk_transcript(transcript)
        
        # Use semantic search to find relevant chunks
        question_embedding = self.embedding_model.encode([question])[0]
        chunk_embeddings = self.embedding_model.encode(chunks)
        
        # Calculate similarities
        similarities = cosine_similarity([question_embedding], chunk_embeddings)[0]
        
        # Get top k most relevant chunks
        top_indices = similarities.argsort()[-top_k:][::-1]
        relevant_chunks = [chunks[i] for i in top_indices]
        
        # Join relevant chunks
        context = "\n".join(relevant_chunks)
        
        # Format prompt for LLM
        prompt = f"""
        Answer the question based ONLY on the following transcript context. Be concise and accurate.
        
        Context from transcript:
        {context}
        
        Question: {question}
        
        Answer:
        """

        try:
            if self.use_vllm and VLLM_AVAILABLE:
                # Use vLLM for faster inference
                sampling_params = SamplingParams(
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                )
                outputs = self.llm.generate(prompt, sampling_params)
                response = outputs[0].outputs[0].text.strip()
            else:
                # Use standard HF pipeline
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    output = self.llm.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                    )
                response = self.tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def start_with_device(self, device_id):
        """Non-blocking start with specified device ID"""
        try:
            device_info = sd.query_devices()[device_id]
            if device_info['max_input_channels'] == 0:
                return False, "Selected device has no input channels!"
                
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
            return True, "Started successfully"
            
        except Exception as e:
            return False, str(e)
    
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
                    answer = self.answer_question_with_hf(user_input)
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
    rag_system = Speech2RAG_HF()
    rag_system.start() 