# import streamlit as st
# import asyncio
# import websockets
# import pyaudio
# import numpy as np
# import wave
# import os
# import time
# import json
# from threading import Thread
# from queue import Queue

# # WebSocket Server URL
# SERVER_URL = "ws://127.0.0.1:8000/"  # Change if needed

# # Audio Configuration
# CHUNK_DURATION = 4  # 5 seconds per chunk
# SAMPLE_RATE = 16000  # Required sample rate
# FORMAT = pyaudio.paInt16  # 16-bit PCM
# CHANNELS = 1  # Mono
# CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)  # Frames per chunk
# SILENCE_PADDING = 0.5  # 0.5s silence at the end
# OVERLAP_DURATION = 0.01  # 1 second overlap between chunks to prevent word loss
# OVERLAP_FRAMES = int(SAMPLE_RATE * OVERLAP_DURATION)

# # Debug directory
# DEBUG_DIR = "debug_audio"
# os.makedirs(DEBUG_DIR, exist_ok=True)

# # Queue for passing transcriptions from WebSocket thread to Streamlit
# transcription_queue = Queue()

# def save_wav(filename, audio_data):
#     """Saves audio data to a WAV file."""
#     filepath = os.path.join(DEBUG_DIR, filename)
#     with wave.open(filepath, "wb") as wf:
#         wf.setnchannels(CHANNELS)
#         wf.setsampwidth(2)  # 16-bit PCM
#         wf.setframerate(SAMPLE_RATE)
#         wf.writeframes(audio_data)
#     print(f"[DEBUG] Saved {filename}")

# async def record_and_stream():
#     """Records audio from the microphone and streams it to the WebSocket server."""
#     audio = pyaudio.PyAudio()

#     # Use a smaller buffer size for more responsive recording
#     buffer_size = 1024  # Smaller buffer for more frequent reads

#     stream = audio.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE,
#                         input=True, frames_per_buffer=buffer_size)

#     async with websockets.connect(SERVER_URL) as websocket:
#         print("[INFO] Connected to server, streaming audio...")

#         chunk_counter = 0
#         buffer = bytearray()  # Use a buffer to accumulate audio data
#         prev_chunk_end = None  # Store the end of the previous chunk for overlap

#         try:
#             while True:
#                 start_time = time.time()

#                 # Keep reading until we have a full chunk
#                 while len(buffer) < CHUNK_SIZE * 2:  # 2 bytes per sample for 16-bit
#                     audio_data = stream.read(buffer_size, exception_on_overflow=False)
#                     buffer.extend(audio_data)

#                 # Extract a full chunk from the buffer
#                 chunk_bytes = buffer[:CHUNK_SIZE * 2]

#                 # Keep overlap for next chunk
#                 if len(buffer) > OVERLAP_FRAMES * 2:
#                     # Remove everything except the overlap portion
#                     buffer = buffer[-(OVERLAP_FRAMES * 2):]
#                 else:
#                     # Clear the buffer if it's smaller than the overlap size
#                     buffer = bytearray()

#                 # Add the overlap from previous chunk if available
#                 if prev_chunk_end is not None and chunk_counter > 0:
#                     final_chunk = prev_chunk_end + chunk_bytes
#                     # Ensure we don't exceed the intended chunk size + overlap
#                     if len(final_chunk) > (CHUNK_SIZE + OVERLAP_FRAMES) * 2:
#                         final_chunk = final_chunk[-(CHUNK_SIZE + OVERLAP_FRAMES) * 2:]
#                 else:
#                     final_chunk = chunk_bytes

#                 # Save the end portion for the next chunk's overlap
#                 prev_chunk_end = chunk_bytes[-OVERLAP_FRAMES * 2:] if len(chunk_bytes) >= OVERLAP_FRAMES * 2 else chunk_bytes

#                 # Save raw chunk for debugging
#                 save_wav(f"raw_chunk_{chunk_counter}.wav", final_chunk)

#                 # Add silence padding at the end
#                 silence = np.zeros(int(SILENCE_PADDING * SAMPLE_RATE), dtype=np.int16).tobytes()
#                 final_audio = final_chunk + silence

#                 # Save final audio (with silence)
#                 save_wav(f"final_chunk_{chunk_counter}.wav", final_audio)

#                 # Send data to server
#                 await websocket.send(final_audio)
#                 print(f"[DEBUG] Sent chunk {chunk_counter} (with overlap and silence)")

#                 # Receive transcription response
#                 try:
#                     response = await asyncio.wait_for(websocket.recv(), timeout=5)
#                     print("[INFO] Received response:", response)

#                     # Parse JSON response and extract text field
#                     try:
#                         response_data = json.loads(response)
#                         if "text" in response_data:
#                             # Extract just the text field from the JSON
#                             transcription_text = response_data["text"]
#                             print("[INFO] Extracted text:", transcription_text)
#                             # Add transcription to queue for Streamlit to display
#                             transcription_queue.put(transcription_text)
#                         else:
#                             print("[WARNING] No 'text' field in response")
#                     except json.JSONDecodeError:
#                         print("[WARNING] Response is not valid JSON, using raw response")
#                         transcription_queue.put(response)

#                 except asyncio.TimeoutError:
#                     print("[WARNING] No response received!")

#                 chunk_counter += 1

#                 # Calculate processing time and adjust sleep to maintain timing
#                 processing_time = time.time() - start_time
#                 sleep_time = max(0, CHUNK_DURATION - OVERLAP_DURATION - processing_time)
#                 if sleep_time > 0:
#                     await asyncio.sleep(sleep_time)
#                 else:
#                     print(f"[WARNING] Processing is taking longer than chunk duration, sleep_time: {sleep_time}")
#                     # Insert minimal sleep to allow other tasks to run
#                     await asyncio.sleep(0.01)

#         except KeyboardInterrupt:
#             print("[INFO] Stopping recording...")

#         except Exception as e:
#             print(f"[ERROR] An error occurred: {e}")

#         finally:
#             stream.stop_stream()
#             stream.close()
#             audio.terminate()

# def run_websocket_client():
#     """Run the WebSocket client in a separate thread"""
#     asyncio.run(record_and_stream())

# # Streamlit app
# def main():
#     st.title("Live Speech Transcription")

#     # Create a placeholder for the transcription
#     transcription_container = st.empty()

#     # Button to start/stop transcription
#     if 'running' not in st.session_state:
#         st.session_state.running = False
#         st.session_state.transcription_text = ""

#     col1, col2 = st.columns(2)

#     with col1:
#         if st.button('Start Transcription' if not st.session_state.running else 'Stop Transcription'):
#             st.session_state.running = not st.session_state.running

#             if st.session_state.running:
#                 # Start the WebSocket client in a separate thread
#                 websocket_thread = Thread(target=run_websocket_client, daemon=True)
#                 websocket_thread.start()
#                 st.session_state.websocket_thread = websocket_thread

#     with col2:
#         if st.button('Clear Transcription'):
#             st.session_state.transcription_text = ""

#     # Display current status
#     st.write(f"Status: {'Recording and transcribing...' if st.session_state.running else 'Idle'}")

#     # Continuous update loop for Streamlit
#     while st.session_state.running:
#         try:
#             # Check if there's new transcription in the queue
#             if not transcription_queue.empty():
#                 new_text = transcription_queue.get(block=False)

#                 # Append new text to existing transcription with a space
#                 if st.session_state.transcription_text:
#                     st.session_state.transcription_text += " " + new_text
#                 else:
#                     st.session_state.transcription_text = new_text

#             # Update the display
#             transcription_container.markdown(f"""
#             ### Transcription:
#             {st.session_state.transcription_text}
#             """)

#             # Short sleep to prevent excessive CPU usage
#             time.sleep(0.1)

#         except Exception as e:
#             st.error(f"Error updating transcription: {e}")
#             break

#     # Always display the current transcription text
#     transcription_container.markdown(f"""
#     ### Transcription:
#     {st.session_state.transcription_text}
#     """)

# if __name__ == "__main__":
#     main()


import streamlit as st
import asyncio
import websockets
import pyaudio
import wave
import os
import time
import json
from threading import Thread
from queue import Queue

# WebSocket Server URL
SERVER_URL = "ws://127.0.0.1:8001/"  # Change if needed

# Audio Configuration
SAMPLE_RATE = 16000  # Required sample rate
FORMAT = pyaudio.paInt16  # 16-bit PCM
CHANNELS = 1  # Mono
CHUNK_DURATION_MS = 100  # 100ms chunks
# ~1600 samples per 100ms
SAMPLES_PER_CHUNK = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)

# Debugging: Save audio chunks
DEBUG_DIR = "debug_audio"
os.makedirs(DEBUG_DIR, exist_ok=True)

# Queue for passing transcriptions from WebSocket thread to Streamlit
transcription_queue = Queue()


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
        print(
            f"[INFO] Sending chunks of {CHUNK_DURATION_MS}ms ({SAMPLES_PER_CHUNK} samples)")

        chunk_counter = 0
        debug_buffer = bytearray()  # Buffer to accumulate 50 chunks for debugging

        try:
            while True:
                # Read exactly 100ms of audio - this blocks until 100ms of audio is captured
                audio_data = stream.read(
                    SAMPLES_PER_CHUNK, exception_on_overflow=False)

                # Add to debug buffer
                debug_buffer.extend(audio_data)

                # Save concatenated chunks every 50 chunks
                # if len(debug_buffer) >= (SAMPLES_PER_CHUNK * 2 * 50):  # 50 chunks of 16-bit audio
                #     save_wav(f"concat_chunks_{chunk_counter-49}_to_{chunk_counter}.wav", debug_buffer)
                #     debug_buffer = bytearray()  # Reset buffer after saving

                # Send the 100ms chunk to the server
                await websocket.send(audio_data)
                # print(f"[DEBUG] Sent chunk {chunk_counter} ({len(audio_data)} bytes)")

                # Check for any server response
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=0.01)
                    # print("[INFO] Server response:", response.text)

                    # Parse JSON response and extract text field
                    try:
                        response_data = json.loads(response)
                        if "text" in response_data:
                            # Extract just the text field from the JSON
                            transcription_text = response_data["text"]
                            print(transcription_text)
                            print("[INFO] Extracted text:", transcription_text)
                            # Add transcription to queue for Streamlit to display
                            transcription_queue.put(transcription_text)
                        else:
                            print("[WARNING] No 'text' field in response")
                    except json.JSONDecodeError:
                        print(
                            "[WARNING] Response is not valid JSON, using raw response")
                        transcription_queue.put(response)

                except asyncio.TimeoutError:
                    pass  # No response received within timeout, continue

                chunk_counter += 1

        except KeyboardInterrupt:
            print("[INFO] Stopping recording...")
            # Save any remaining audio in the debug buffer
            if debug_buffer:
                save_wav(
                    f"final_concat_chunks_to_{chunk_counter}.wav", debug_buffer)

        except Exception as e:
            print(f"[ERROR] An error occurred: {e}")

        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()


def run_websocket_client():
    """Run the WebSocket client in a separate thread"""
    asyncio.run(record_and_stream())

# Streamlit app


def main():
    st.title("Real-time Speech Transcription")

    # Create a placeholder for the transcription
    transcription_container = st.empty()

    # Initialize session state variables
    if 'running' not in st.session_state:
        st.session_state.running = False
        st.session_state.transcription_text = ""

    col1, col2 = st.columns(2)

    with col1:
        if st.button('Start Transcription' if not st.session_state.running else 'Stop Transcription'):
            st.session_state.running = not st.session_state.running

            if st.session_state.running:
                # Start the WebSocket client in a separate thread
                websocket_thread = Thread(
                    target=run_websocket_client, daemon=True)
                websocket_thread.start()
                st.session_state.websocket_thread = websocket_thread

    with col2:
        if st.button('Clear Transcription'):
            st.session_state.transcription_text = ""

    # Display current status
    st.write(
        f"Status: {'Recording and transcribing...' if st.session_state.running else 'Idle'}")

    # Create a container for displaying the transcription with styling
    st.markdown("""
    <style>
    .transcription-box {
        padding: 20px;
        border-radius: 5px;
        border: 1px solid #ddd;
        background-color: #f9f9f9;
        line-height: 1.5;
        min-height: 200px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Continuous update loop for Streamlit
    while st.session_state.running:
        try:
            # Check if there's new transcription in the queue
            if not transcription_queue.empty():
                new_text = transcription_queue.get(block=False)

                # Append new text to existing transcription with a space
                if st.session_state.transcription_text:
                    st.session_state.transcription_text += " " + new_text
                else:
                    st.session_state.transcription_text = new_text

            # Update the display
            transcription_container.markdown(f"""
            ### Transcription:
            <div class="transcription-box">
            {st.session_state.transcription_text}
            </div>
            """, unsafe_allow_html=True)

            # Short sleep to prevent excessive CPU usage
            time.sleep(0.1)

        except Exception as e:
            st.error(f"Error updating transcription: {e}")
            break

    # Always display the current transcription text
    transcription_container.markdown(f"""
    ### Transcription:
    <div class="transcription-box">
    {st.session_state.transcription_text}
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
