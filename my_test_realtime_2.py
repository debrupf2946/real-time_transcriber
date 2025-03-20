import asyncio
import websockets
import pyaudio
import numpy as np
import wave
import os
import time

# WebSocket Server URL
SERVER_URL = "ws://127.0.0.1:8000/"  # Change if needed

# Audio Configuration
CHUNK_DURATION = 5  # 5 seconds per chunk
SAMPLE_RATE = 16000  # Required sample rate
FORMAT = pyaudio.paInt16  # 16-bit PCM
CHANNELS = 1  # Mono
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)  # Frames per chunk
SILENCE_PADDING = 0.5  # 0.5s silence at the end
OVERLAP_DURATION = 1  # 1 second overlap between chunks to prevent word loss
OVERLAP_FRAMES = int(SAMPLE_RATE * OVERLAP_DURATION)

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
    
    # Use a smaller buffer size for more responsive recording
    buffer_size = 1024  # Smaller buffer for more frequent reads
    
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE,
                        input=True, frames_per_buffer=buffer_size)

    async with websockets.connect(SERVER_URL) as websocket:
        print("[INFO] Connected to server, streaming audio...")

        chunk_counter = 0
        buffer = bytearray()  # Use a buffer to accumulate audio data
        prev_chunk_end = None  # Store the end of the previous chunk for overlap
        
        try:
            while True:
                start_time = time.time()
                
                # Keep reading until we have a full chunk
                while len(buffer) < CHUNK_SIZE * 2:  # 2 bytes per sample for 16-bit
                    audio_data = stream.read(buffer_size, exception_on_overflow=False)
                    buffer.extend(audio_data)
                
                # Extract a full chunk from the buffer
                chunk_bytes = buffer[:CHUNK_SIZE * 2]
                
                # Keep overlap for next chunk
                if len(buffer) > OVERLAP_FRAMES * 2:
                    # Remove everything except the overlap portion
                    buffer = buffer[-(OVERLAP_FRAMES * 2):]
                else:
                    # Clear the buffer if it's smaller than the overlap size
                    buffer = bytearray()
                
                # Add the overlap from previous chunk if available
                if prev_chunk_end is not None and chunk_counter > 0:
                    final_chunk = prev_chunk_end + chunk_bytes
                    # Ensure we don't exceed the intended chunk size + overlap
                    if len(final_chunk) > (CHUNK_SIZE + OVERLAP_FRAMES) * 2:
                        final_chunk = final_chunk[-(CHUNK_SIZE + OVERLAP_FRAMES) * 2:]
                else:
                    final_chunk = chunk_bytes
                
                # Save the end portion for the next chunk's overlap
                prev_chunk_end = chunk_bytes[-OVERLAP_FRAMES * 2:] if len(chunk_bytes) >= OVERLAP_FRAMES * 2 else chunk_bytes
                
                # Save raw chunk for debugging
                save_wav(f"raw_chunk_{chunk_counter}.wav", final_chunk)
                
                # Add silence padding at the end
                silence = np.zeros(int(SILENCE_PADDING * SAMPLE_RATE), dtype=np.int16).tobytes()
                final_audio = final_chunk + silence
                
                # Save final audio (with silence)
                save_wav(f"final_chunk_{chunk_counter}.wav", final_audio)
                
                # Send data to server
                await websocket.send(final_audio)
                print(f"[DEBUG] Sent chunk {chunk_counter} (with overlap and silence)")
                
                # Receive transcription response
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5)
                    print("[INFO] Transcription:", response)
                except asyncio.TimeoutError:
                    print("[WARNING] No response received!")
                
                chunk_counter += 1
                
                # Calculate processing time and adjust sleep to maintain timing
                processing_time = time.time() - start_time
                sleep_time = max(0, CHUNK_DURATION - OVERLAP_DURATION - processing_time)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                else:
                    print(f"[WARNING] Processing is taking longer than chunk duration, sleep_time: {sleep_time}")
                    # Insert minimal sleep to allow other tasks to run
                    await asyncio.sleep(0.01)
                
        except KeyboardInterrupt:
            print("[INFO] Stopping recording...")
        
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