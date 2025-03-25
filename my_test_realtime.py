import asyncio
import websockets
import pyaudio
import numpy as np
import wave
import io

# WebSocket Server URL
SERVER_URL = "ws://localhost:8000/"  # Change if needed

# Audio Configuration
CHUNK_DURATION = 5  # 3 seconds per chunk
SAMPLE_RATE = 16000  # Required sample rate
FORMAT = pyaudio.paInt16  # 16-bit PCM
CHANNELS = 1  # Mono
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)  # Frames per chunk
SILENCE_PADDING = 0.0001  # 0.5s silence to signal chunk completion

def convert_audio(audio_data):
    """
    Ensures audio is 16kHz, 16-bit PCM, Mono.
    Returns correctly formatted audio as bytes.
    """
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(16000)  # 16kHz
        wf.writeframes(audio_data)

    return buffer.getvalue()

async def record_and_stream():
    """
    Records audio from the microphone and streams it to the WebSocket server.
    """
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE,
                        input=True, frames_per_buffer=CHUNK_SIZE)

    async with websockets.connect(SERVER_URL) as websocket:
        print("[INFO] Connected to server, streaming audio...")

        try:
            while True:
                # Read audio chunk
                audio_data = stream.read(CHUNK_SIZE, exception_on_overflow=False)

                # Convert to required format
                audio_bytes = convert_audio(audio_data)

                # Add silence at the end of the chunk
                silence = np.zeros(int(SILENCE_PADDING * SAMPLE_RATE), dtype=np.int16).tobytes()
                final_audio = audio_bytes + silence

                # Send data to server
                await websocket.send(final_audio)
                print(f"[DEBUG] Sent {len(final_audio)} bytes")

                # Receive transcription response
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5)
                    print("[INFO] Transcription:", response)
                except asyncio.TimeoutError:
                    print("[WARNING] No response received!")

                await asyncio.sleep(0.1)  # Simulate real-time streaming

        except KeyboardInterrupt:
            print("[INFO] Stopping recording...")

        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()

# Run the recording and streaming function
# asyncio.run(record_and_stream())