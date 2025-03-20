import asyncio
import websockets
import pyaudio
import wave
import os
import time

# WebSocket Server URL
SERVER_URL = "ws://127.0.0.1:8000/"  # Change if needed

# Audio Configuration
SAMPLE_RATE = 16000  # Required sample rate
FORMAT = pyaudio.paInt16  # 16-bit PCM
CHANNELS = 1  # Mono
CHUNK_DURATION_MS = 100  # 100ms chunks
SAMPLES_PER_CHUNK = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)  # ~1600 samples per 100ms

# Debugging: Save audio chunks
DEBUG_DIR = "debug_audio"
os.makedirs(DEBUG_DIR, exist_ok=True)

def save_wav(filename, audio_data):
    """Saves audio data to a WAV file."""
    filepath = os.path.join(DEBUG_DIR, filename)
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data)
    print(f"[DEBUG] Saved {filename}")

async def record_and_stream():
    """Records audio from the microphone and streams it to the WebSocket server."""
    audio = pyaudio.PyAudio()
    
    # Set buffer size to exactly match 100ms of audio
    stream = audio.open(format=FORMAT, 
                        channels=CHANNELS, 
                        rate=SAMPLE_RATE,
                        input=True, 
                        frames_per_buffer=SAMPLES_PER_CHUNK)

    async with websockets.connect(SERVER_URL) as websocket:
        print("[INFO] Connected to server, streaming audio...")
        print(f"[INFO] Sending chunks of {CHUNK_DURATION_MS}ms ({SAMPLES_PER_CHUNK} samples)")

        chunk_counter = 0
        debug_buffer = bytearray()  # Buffer to accumulate 50 chunks for debugging
        
        try:
            while True:
                # Read exactly 100ms of audio - this blocks until 100ms of audio is captured
                audio_data = stream.read(SAMPLES_PER_CHUNK, exception_on_overflow=False)
                
                # Add to debug buffer
                debug_buffer.extend(audio_data)
                
                # Save concatenated chunks every 50 chunks
                if len(debug_buffer) >= (SAMPLES_PER_CHUNK * 2 * 50):  # 50 chunks of 16-bit audio
                    save_wav(f"concat_chunks_{chunk_counter-49}_to_{chunk_counter}.wav", debug_buffer)
                    debug_buffer = bytearray()  # Reset buffer after saving

                # Send the 100ms chunk to the server
                await websocket.send(audio_data)
                print(f"[DEBUG] Sent chunk {chunk_counter} ({len(audio_data)} bytes)")
                
                # Optional: Check for any server response
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=0.01)
                    print("[INFO] Server response:", response)
                except asyncio.TimeoutError:
                    pass
                
                chunk_counter += 1
                
        except KeyboardInterrupt:
            print("[INFO] Stopping recording...")
            # Save any remaining audio in the debug buffer
            if debug_buffer:
                save_wav(f"final_concat_chunks_to_{chunk_counter}.wav", debug_buffer)
        
        except Exception as e:
            print(f"[ERROR] An error occurred: {e}")
            
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()

# Run the recording and streaming function
if __name__ == "__main__":
    print("[INFO] Starting audio streaming. Press Ctrl+C to stop.")
    asyncio.run(record_and_stream())