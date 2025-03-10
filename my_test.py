import asyncio
import websockets
import wave
import io
import numpy as np
import soundfile as sf
import librosa


SERVER_URL = "ws://localhost:8000/"
AUDIO_FILE = "recording_20250226_130752.wav" 

def convert_audio(file_path):
    """
    Ensures the audio is in the correct format (16kHz, 16-bit PCM, mono).
    Returns the correctly formatted audio as bytes.
    """
    with sf.SoundFile(file_path) as sf_audio:
        sample_rate = sf_audio.samplerate
        sample_width = 2  # 16-bit PCM
        num_channels = sf_audio.channels
        audio_data = sf_audio.read(dtype='int16')  # Convert to 16-bit PCM

        # Convert stereo to mono if needed
        if num_channels > 1:
            audio_data = np.mean(audio_data, axis=1, dtype='int16')

        # Resample if needed
        if sample_rate != 16000:
            audio_data = librosa.resample(audio_data.astype(np.float32), orig_sr=sample_rate, target_sr=16000).astype(np.int16)

    # Write to an in-memory WAV buffer
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(sample_width)  # 16-bit
        wf.setframerate(16000)  # 16kHz
        wf.writeframes(audio_data.tobytes())

    return buffer.getvalue()

async def test_transcription():
    async with websockets.connect(SERVER_URL) as websocket:
        print("[INFO] Connected to server")

        # Convert audio if necessary
        audio_bytes = convert_audio(AUDIO_FILE)

        # Send audio data
        await websocket.send(audio_bytes)
        print("[INFO] Audio data sent!")

        # Receive response
        response = await websocket.recv()
        print("[INFO] Transcription Response:", response)

# Run the test
asyncio.run(test_transcription())

 # Make sure this file exists and is in the correct format

