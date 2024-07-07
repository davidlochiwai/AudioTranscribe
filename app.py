import streamlit as st
import moviepy.editor as mp
import os
from tempfile import NamedTemporaryFile
import utils
from openai import AzureOpenAI

# Login Page
utils.setup_page("Transcription App")

# Setup client and API parameters
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
client = AzureOpenAI(
    api_key=api_key,
    api_version="2024-02-01",
    azure_endpoint=endpoint
)

MAX_FILE_SIZE = 26214400  # in bytes, approximately 26.21 MB

def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        result = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper"
        )
    return result.text

def handle_audio_file(file_path, is_audio):
    if is_audio:
        return file_path  # Return the audio file path directly
    else:
        # Extract audio from video
        video = mp.VideoFileClip(file_path)
        audio_path = "temp_audio.wav"
        video.audio.write_audiofile(audio_path, fps=8000, codec='pcm_s16le')
        video.close()
        safe_remove(file_path)  # Remove the temporary video file safely
        return audio_path

def split_audio_by_size(audio_path, max_size):
    if os.path.getsize(audio_path) <= max_size:
        return [audio_path]  # Return the original file as a single chunk if it's not too large

    audio = mp.AudioFileClip(audio_path)
    total_duration = audio.duration
    estimated_chunk_length = total_duration * (max_size / os.path.getsize(audio_path))

    chunks = []
    start = 0
    progress_bar = st.progress(0)
    st.text("Splitting audio into manageable chunks...")
    while start < total_duration:
        end = min(start + estimated_chunk_length, total_duration)
        chunk_path = f"temp_chunk_{int(start)}.wav"
        audio.subclip(start, end).write_audiofile(chunk_path, codec='pcm_s16le')
        if os.path.getsize(chunk_path) > max_size:
            safe_remove(chunk_path)  # Remove the chunk safely if it's too large
            estimated_chunk_length *= 0.9  # Reduce the chunk size if it's too large
        else:
            chunks.append(chunk_path)
            start = end
        progress_bar.progress(min(1, start / total_duration))
    audio.close()
    progress_bar.empty()  # Remove progress bar after operation
    st.text("")  # Clear text
    return chunks

def safe_remove(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

def app():
    st.title("Audio Transcription from Video or Audio")
    uploaded_file = st.file_uploader("Upload a video or audio file", type=["mp4", "avi", "mov", "mp3", "wav", "aac"])
    
    if uploaded_file:
        if st.button("Transcribe Audio"):
            file_type = uploaded_file.name.split('.')[-1]
            is_audio = file_type in ['mp3', 'wav', 'aac']
            
            with NamedTemporaryFile(delete=False, suffix=f'.{file_type}') as temp_file:
                temp_file.write(uploaded_file.getvalue())
                file_path = temp_file.name
            
            audio_path = handle_audio_file(file_path, is_audio)
            progress_bar = st.progress(0)
            st.text("Preparing audio for transcription...")
            if os.path.getsize(audio_path) > MAX_FILE_SIZE:
                chunk_files = split_audio_by_size(audio_path, MAX_FILE_SIZE)
            else:
                chunk_files = [audio_path]  # Use the audio file directly if it's within the size limit
            
            transcriptions = []
            for i, chunk in enumerate(chunk_files):
                st.text(f"Transcribing chunk {i+1} of {len(chunk_files)}...")
                transcription = transcribe_audio(chunk)
                transcriptions.append(transcription)
                progress_bar.progress((i + 1) / len(chunk_files))
            
            final_transcription = " ".join(transcriptions)
            transcription_area = st.text_area("Transcription", final_transcription, height=300)
            st.download_button(
                label="Download Transcription",
                data=final_transcription,
                file_name="transcription.txt",
                mime="text/plain"
            )
            progress_bar.empty()  # Remove progress bar after transcription
            st.text("")  # Clear text
            
            # Clean up
            safe_remove(audio_path)
            for chunk in chunk_files:
                safe_remove(chunk)

if __name__ == "__main__":
    app()







