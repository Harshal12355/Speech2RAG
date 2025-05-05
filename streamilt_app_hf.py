import streamlit as st
import sounddevice as sd
import numpy as np
import threading
import time
import queue
import os
import sys
import torch
from speech_rag_hf import Speech2RAG_HF, VLLM_AVAILABLE

# Initialize session state
if "transcription_active" not in st.session_state:
    st.session_state.transcription_active = False
    st.session_state.rag_system = None
    st.session_state.transcript = ""
    st.session_state.transcript_history = []
    st.session_state.audio_queue = queue.Queue()
    st.session_state.audio_device = None
    st.session_state.questions = []
    st.session_state.answers = []

# Streamlit UI
st.title("🎙️ Speech2RAG: Real-Time Transcription + RAG (HF/vLLM Version)")
st.markdown("""
- Transcribe audio in real-time with VOSK
- Ask questions about the transcription using Hugging Face models
- Powered by vLLM for faster inference
""")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    
    # List available audio devices
    devices = sd.query_devices()
    device_options = []
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:  # Only show input devices
            device_options.append(f"{i}: {device['name']}")
    
    selected_device = st.selectbox(
        "Audio Input Device",
        options=device_options,
        index=0 if device_options else None,
    )
    
    # Get device ID from selection
    device_id = int(selected_device.split(":")[0]) if selected_device else None
    
    model_path = st.selectbox(
        "VOSK Model",
        options=["models/vosk-model-small-en-us-0.15", "models/vosk-model-en-us-0.22", "model/vosk-model-en-us-0.42-gigaspeech"],
        index=0,
    )
    
    # Hugging Face model options
    hf_model = st.selectbox(
        "Hugging Face Model",
        options=[
            "google/gemma-1.1-2b-it", 
            "google/gemma-2-9b", 
            "mistralai/Mistral-7B-Instruct-v0.2",
            "meta-llama/Llama-2-7b-chat-hf"
        ],
        index=0,
    )
    
    # vLLM acceleration
    use_vllm = st.checkbox("Use vLLM acceleration", value=VLLM_AVAILABLE, disabled=not VLLM_AVAILABLE)
    
    if not VLLM_AVAILABLE:
        st.warning("vLLM is not available on your system. Using standard HuggingFace inference.")
    
    # Device selection
    compute_device = st.radio(
        "Compute Device",
        options=["cuda" if torch.cuda.is_available() else "cpu", "cpu"],
        index=0,
    )
    
    # Embedding model
    embedding_model = st.selectbox(
        "Embedding Model",
        options=[
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "hkunlp/instructor-xl"
        ],
        index=0,
    )

# Start/Stop Transcription
col1, col2 = st.columns(2)
with col1:
    if st.button("🎤 Start Transcription", disabled=st.session_state.transcription_active):
        with st.spinner("Loading speech recognition model..."):
            st.session_state.rag_system = Speech2RAG_HF(
                model_path=model_path, 
                hf_model_name=hf_model,
                embedding_model_name=embedding_model,
                use_vllm=use_vllm,
                device=compute_device
            )
            success, message = st.session_state.rag_system.start_with_device(device_id)
            
            if success:
                st.session_state.transcription_active = True
                st.success("Transcription started! Speak now...")
            else:
                st.error(f"Failed to start: {message}")

with col2:
    if st.button("⏹️ Stop Transcription", disabled=not st.session_state.transcription_active):
        if st.session_state.rag_system:
            st.session_state.rag_system.stop()
            st.session_state.transcription_active = False
            st.warning("Transcription stopped.")

# Display live transcript in a container
st.subheader("Live Transcript")
transcript_container = st.container()
transcript_area = transcript_container.empty()

# Update transcript in a container
if st.session_state.transcription_active and st.session_state.rag_system:
    # This will create placeholder for transcript
    transcript_text = st.session_state.rag_system.get_full_transcript() or "Waiting for speech..."
    transcript_area.text_area("", value=transcript_text, height=200, key="transcript_display", disabled=True)

# Question and answer section
st.subheader("Ask Questions")
question = st.text_input("Ask a question about the transcription:", key="question_input")

if st.button("Submit Question", disabled=not st.session_state.transcription_active) and question:
    if not st.session_state.rag_system:
        st.error("Transcription system is not active.")
    else:
        with st.spinner("Generating answer..."):
            # Use the new answer_question_with_hf method
            answer = st.session_state.rag_system.answer_question_with_hf(
                question,
                top_k=3,
                max_new_tokens=512,
                temperature=0.3
            )
            st.session_state.questions.append(question)
            st.session_state.answers.append(answer)

# Display Q&A history
if st.session_state.questions:
    st.subheader("Question & Answer History")
    for i, (q, a) in enumerate(zip(st.session_state.questions, st.session_state.answers)):
        with st.expander(f"Q: {q}", expanded=(i == len(st.session_state.questions) - 1)):
            st.markdown(a)

# Auto-refresh the transcript (updates every 2 seconds)
if st.session_state.transcription_active:
    st.empty()  # This is required for the refresh to work
    time.sleep(2)  # Wait for 2 seconds
    st.rerun()  # Updated from st.experimental_rerun()

# Instructions
with st.expander("How to Use"):
    st.markdown("""
    1. Select your **audio input device** in the sidebar
    2. Choose the **VOSK model** and **Hugging Face model**
    3. Toggle **vLLM acceleration** if available
    4. Select appropriate **computing device** (CPU/GPU)
    5. Click **Start Transcription** and speak
    6. Ask questions in the input box
    7. Click **Stop Transcription** when done
    
    **Note**: For system audio, select "Stereo Mix" or similar device if available.
    """)

# System info
with st.expander("System Information"):
    st.markdown(f"""
    - Python version: {sys.version.split()[0]}
    - Torch version: {torch.__version__}
    - CUDA available: {torch.cuda.is_available()}
    - GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}
    """) 