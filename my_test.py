import asyncio
import websockets
import wave
import io
import json
import os
import numpy as np
import soundfile as sf
import librosa
import time  # Import time module for measurement

SERVER_URL = "ws://localhost:8000/"
AUDIO_FILE = "debug/debug_audio.wav"  
DEBUG_DIR = "debug2"
DEBUG_AUDIO_FILE = os.path.join(DEBUG_DIR, "debug_audio.wav")  # Save in debug directory
TIMEOUT_SECONDS = 20  # Timeout duration for WebSocket response

def save_audio_for_debugging(audio_data, file_name=DEBUG_AUDIO_FILE):
    """
    Saves the processed audio data as a WAV file for debugging.
    """
    os.makedirs(DEBUG_DIR, exist_ok=True)
    
    try:
        with wave.open(file_name, "wb") as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit PCM
            wf.setframerate(16000)  # 16kHz
            wf.writeframes(audio_data.tobytes())
        print(f"[DEBUG] Saved audio for debugging: {file_name}")
    except Exception as e:
        print(f"[ERROR] Failed to save debug audio: {e}")

def convert_audio(file_path):
    """
    Ensures the audio is in the correct format (16kHz, 16-bit PCM, mono).
    Appends 0.6 seconds of silence at the end.
    Returns the correctly formatted audio as bytes.
    """
    with sf.SoundFile(file_path) as sf_audio:
        sample_rate = sf_audio.samplerate
        num_channels = sf_audio.channels
        audio_data = sf_audio.read(dtype='int16')  # Convert to 16-bit PCM

        # Convert stereo to mono if needed
        if num_channels > 1:
            audio_data = np.mean(audio_data, axis=1, dtype='int16')

        # Resample if needed
        if sample_rate != 16000:
            audio_data = librosa.resample(audio_data.astype(np.float32), orig_sr=sample_rate, target_sr=16000).astype(np.int16)

    # Generate 2 seconds of silence (16kHz, 16-bit PCM)
    silence_duration = 2  # seconds
    silence_samples = int(16000 * silence_duration)  # 16000 samples per second
    silence = np.zeros(silence_samples, dtype='int16')

    # Append silence to the audio data
    modified_audio = np.concatenate((audio_data, silence))  # Append at the end

    # Write to an in-memory WAV buffer
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(16000)  # 16kHz
        wf.writeframes(modified_audio.tobytes())

    return buffer.getvalue()

async def whisper_transcription(audio_file=AUDIO_FILE):
    async with websockets.connect(SERVER_URL) as websocket:
        print("[INFO] Connected to server")

        # Convert audio if necessary
        audio_bytes = convert_audio(audio_file)

        # Measure time to send
        send_start_time = time.time()
        await websocket.send(audio_bytes)
        send_end_time = time.time()
        print(f"[INFO] Audio data sent! Time taken: {send_end_time - send_start_time:.3f} seconds")

        # Measure latency (round-trip)
        latency_start_time = time.time()
        
        try:
            response = await asyncio.wait_for(websocket.recv(), timeout=TIMEOUT_SECONDS)
            print(response)
            latency_end_time = time.time()

            response_data = json.loads(response)
            transcription_text = response_data["text"]
            print(f"[INFO] Transcription Response: {transcription_text}")
            print(f"[INFO] Total Latency: {latency_end_time - latency_start_time:.3f} seconds")

            return transcription_text, latency_end_time - latency_start_time

        except asyncio.TimeoutError:
            print(f"[ERROR] No response from server within {TIMEOUT_SECONDS} seconds.")
            return None, None

# Run the test
asyncio.run(whisper_transcription())





#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# import asyncio
# import websockets
# import wave
# import io
# import json
# import os
# import numpy as np
# import soundfile as sf
# import librosa
# import time  # Import time module for measurement

# SERVER_URL = "ws://localhost:8000/"
# # AUDIO_FILE = "recording_20250226_130752.wav"  
# AUDIO_FILE = "debug/debug_audio.wav"  
# DEBUG_DIR = "debug2"
# DEBUG_AUDIO_FILE = os.path.join(DEBUG_DIR, "debug_audio.wav")  # Save in debug directory

# def save_audio_for_debugging(audio_data, file_name=DEBUG_AUDIO_FILE):
#     """
#     Saves the processed audio data as a WAV file for debugging.
#     """
#     os.makedirs(DEBUG_DIR, exist_ok=True)
    
#     try:
#         with wave.open(file_name, "wb") as wf:
#             wf.setnchannels(1)  # Mono
#             wf.setsampwidth(2)  # 16-bit PCM
#             wf.setframerate(16000)  # 16kHz
#             wf.writeframes(audio_data.tobytes())
#         print(f"[DEBUG] Saved audio for debugging: {file_name}")
#     except Exception as e:
#         print(f"[ERROR] Failed to save debug audio: {e}")

# def convert_audio(file_path):
#     """
#     Ensures the audio is in the correct format (16kHz, 16-bit PCM, mono).
#     Appends 0.6 seconds of silence at the end.
#     Returns the correctly formatted audio as bytes.
#     """
#     with sf.SoundFile(file_path) as sf_audio:
#         sample_rate = sf_audio.samplerate
#         num_channels = sf_audio.channels
#         audio_data = sf_audio.read(dtype='int16')  # Convert to 16-bit PCM

#         # Convert stereo to mono if needed
#         if num_channels > 1:
#             audio_data = np.mean(audio_data, axis=1, dtype='int16')

#         # Resample if needed
#         if sample_rate != 16000:
#             audio_data = librosa.resample(audio_data.astype(np.float32), orig_sr=sample_rate, target_sr=16000).astype(np.int16)

#     # Generate 0.6 seconds of silence (16kHz, 16-bit PCM)
#     silence_duration = 2  # seconds
#     silence_samples = int(16000 * silence_duration)  # 16000 samples per second
#     silence = np.zeros(silence_samples, dtype='int16')

#     # Append silence to the audio data
#     modified_audio = np.concatenate((audio_data, silence))  # Append at the end

#     # Save debug audio
#     # save_audio_for_debugging(modified_audio, DEBUG_AUDIO_FILE)

#     # Write to an in-memory WAV buffer
#     buffer = io.BytesIO()
#     with wave.open(buffer, "wb") as wf:
#         wf.setnchannels(1)  # Mono
#         wf.setsampwidth(2)  # 16-bit PCM
#         wf.setframerate(16000)  # 16kHz
#         wf.writeframes(modified_audio.tobytes())

#     return buffer.getvalue()

# async def whisper_transcription(audio_file=AUDIO_FILE):
#     async with websockets.connect(SERVER_URL) as websocket:
#         print("[INFO] Connected to server")

#         # Convert audio if necessary
#         audio_bytes = convert_audio(audio_file)

#         # Measure time to send
#         send_start_time = time.time()
#         await websocket.send(audio_bytes)
#         send_end_time = time.time()
#         print(f"[INFO] Audio data sent! Time taken: {send_end_time - send_start_time:.3f} seconds")

#         # Measure latency (round-trip)
#         latency_start_time = time.time()
#         response = await websocket.recv()
#         latency_end_time = time.time()
#         response_data = json.loads(response)
#         transcription_text = response_data["text"]
#         print(f"[INFO] Transcription Response: {transcription_text}")
#         print(f"[INFO] Total Latency: {latency_end_time - latency_start_time:.3f} seconds")

#         latency = latency_end_time - latency_start_time
        
#         return transcription_text, latency

# # Run the test
# asyncio.run(whisper_transcription())



# import asyncio
# import websockets
# import wave
# import io
# import json
# import os
# import numpy as np
# import soundfile as sf
# import librosa
# import time  # Import time module for measurement

# SERVER_URL = "ws://localhost:8000/"
# # AUDIO_FILE = "recording_20250226_130752.wav"  
# AUDIO_FILE = "audio_chunks/chunk_2.wav"  
# DEBUG_DIR = "debug2"
# DEBUG_AUDIO_FILE = os.path.join(DEBUG_DIR, "debug_audio.wav")  # Save in debug directory

# def save_audio_for_debugging(audio_data, file_name=DEBUG_AUDIO_FILE):
#     """
#     Saves the processed audio data as a WAV file for debugging.
#     """
#     # Ensure debug directory exists
#     os.makedirs(DEBUG_DIR, exist_ok=True)
    
#     try:
#         with wave.open(file_name, "wb") as wf:
#             wf.setnchannels(1)  # Mono
#             wf.setsampwidth(2)  # 16-bit PCM
#             wf.setframerate(16000)  # 16kHz
#             wf.writeframes(audio_data.tobytes())
#         print(f"[DEBUG] Saved audio for debugging: {file_name}")
#     except Exception as e:
#         print(f"[ERROR] Failed to save debug audio: {e}")

# def convert_audio(file_path):
#     """
#     Ensures the audio is in the correct format (16kHz, 16-bit PCM, mono).
#     Returns the correctly formatted audio as bytes.
#     """
#     with sf.SoundFile(file_path) as sf_audio:
#         sample_rate = sf_audio.samplerate
#         num_channels = sf_audio.channels
#         audio_data = sf_audio.read(dtype='int16')  # Convert to 16-bit PCM

#         # Convert stereo to mono if needed
#         if num_channels > 1:
#             audio_data = np.mean(audio_data, axis=1, dtype='int16')

#         # Resample if needed
#         if sample_rate != 16000:
#             audio_data = librosa.resample(audio_data.astype(np.float32), orig_sr=sample_rate, target_sr=16000).astype(np.int16)

#     # Save debug audio
#     save_audio_for_debugging(audio_data, DEBUG_AUDIO_FILE)

#     # Write to an in-memory WAV buffer
#     buffer = io.BytesIO()
#     with wave.open(buffer, "wb") as wf:
#         wf.setnchannels(1)  # Mono
#         wf.setsampwidth(2)  # 16-bit PCM
#         wf.setframerate(16000)  # 16kHz
#         wf.writeframes(audio_data.tobytes())

#     return buffer.getvalue()

# async def whisper_transcription(audio_file=AUDIO_FILE):
#     async with websockets.connect(SERVER_URL) as websocket:
#         print("[INFO] Connected to server")

#         # Convert audio if necessary
#         audio_bytes = convert_audio(audio_file)

#         # Measure time to send
#         send_start_time = time.time()
#         await websocket.send(audio_bytes)
#         send_end_time = time.time()
#         print(f"[INFO] Audio data sent! Time taken: {send_end_time - send_start_time:.3f} seconds")

#         # Measure latency (round-trip)
#         latency_start_time = time.time()
#         response = await websocket.recv()
#         latency_end_time = time.time()
#         response_data = json.loads(response)
#         transcription_text = response_data["text"]
#         print(f"[INFO] Transcription Response: {transcription_text}")
#         print(f"[INFO] Total Latency: {latency_end_time - latency_start_time:.3f} seconds")

#         latency = latency_end_time - latency_start_time
        
#         return transcription_text, latency

# # Run the test
# asyncio.run(whisper_transcription())




# import asyncio
# import websockets
# import wave
# import io
# import json
# import numpy as np
# import soundfile as sf
# import librosa
# import time  # Import time module for measurement

# SERVER_URL = "ws://localhost:8000/"
# AUDIO_FILE = "recording_20250226_130752.wav" 

# def convert_audio(file_path):
#     """
#     Ensures the audio is in the correct format (16kHz, 16-bit PCM, mono).
#     Returns the correctly formatted audio as bytes.
#     """
#     with sf.SoundFile(file_path) as sf_audio:
#         sample_rate = sf_audio.samplerate
#         sample_width = 2  # 16-bit PCM
#         num_channels = sf_audio.channels
#         audio_data = sf_audio.read(dtype='int16')  # Convert to 16-bit PCM

#         # Convert stereo to mono if needed
#         if num_channels > 1:
#             audio_data = np.mean(audio_data, axis=1, dtype='int16')

#         # Resample if needed
#         if sample_rate != 16000:
#             audio_data = librosa.resample(audio_data.astype(np.float32), orig_sr=sample_rate, target_sr=16000).astype(np.int16)

#     # Write to an in-memory WAV buffer
#     buffer = io.BytesIO()
#     with wave.open(buffer, "wb") as wf:
#         wf.setnchannels(1)  # Mono
#         wf.setsampwidth(sample_width)  # 16-bit
#         wf.setframerate(16000)  # 16kHz
#         wf.writeframes(audio_data.tobytes())

#     return buffer.getvalue()

# async def whisper_transcription(audio_file=AUDIO_FILE):
#     async with websockets.connect(SERVER_URL) as websocket:
#         print("[INFO] Connected to server")

#         # Convert audio if necessary
#         audio_bytes = convert_audio(audio_file)

#         # Measure time to send
#         send_start_time = time.time()
#         await websocket.send(audio_bytes)
#         send_end_time = time.time()
#         print(f"[INFO] Audio data sent! Time taken: {send_end_time - send_start_time:.3f} seconds")

#         # Measure latency (round-trip)
#         latency_start_time = time.time()
#         response = await websocket.recv()
#         latency_end_time = time.time()
#         response_data = json.loads(response)
#         transcription_text = response_data["text"]
#         print(f"[INFO] Transcription Response: {transcription_text}")
#         print(f"[INFO] Total Latency: {latency_end_time - latency_start_time:.3f} seconds")
        
#         latency=latency_end_time - latency_start_time
        
#         return transcription_text,latency

# # Run the test
# asyncio.run(whisper_transcription())




# import asyncio
# import websockets
# import wave
# import io
# import numpy as np
# import soundfile as sf
# import librosa
# import time
# import json

# SERVER_URL = "ws://localhost:8000/"
# AUDIO_FILE = "recording_20250226_130752.wav"

# def convert_audio(file_path):
#     """
#     Ensures the audio is in the correct format (16kHz, 16-bit PCM, mono).
#     Returns the correctly formatted audio as bytes.
#     """
#     with sf.SoundFile(file_path) as sf_audio:
#         sample_rate = sf_audio.samplerate
#         sample_width = 2  # 16-bit PCM
#         num_channels = sf_audio.channels
#         audio_data = sf_audio.read(dtype='int16')  # Convert to 16-bit PCM

#         # Convert stereo to mono if needed
#         if num_channels > 1:
#             audio_data = np.mean(audio_data, axis=1, dtype='int16')

#         # Resample if needed
#         if sample_rate != 16000:
#             audio_data = librosa.resample(audio_data.astype(np.float32), orig_sr=sample_rate, target_sr=16000).astype(np.int16)

#     # Write to an in-memory WAV buffer
#     buffer = io.BytesIO()
#     with wave.open(buffer, "wb") as wf:
#         wf.setnchannels(1)  # Mono
#         wf.setsampwidth(sample_width)  # 16-bit
#         wf.setframerate(16000)  # 16kHz
#         wf.writeframes(audio_data.tobytes())

#     return buffer.getvalue()

# async def test_transcription():
#     async with websockets.connect(SERVER_URL) as websocket:
#         print("[INFO] Connected to server")

#         # Send configuration first
#         config = {
#             "type": "config",
#             "data": {
#                 "sampling_rate": 16000,
#                 "save_audio": True
#             }
#         }
#         await websocket.send(json.dumps(config))
#         print("[INFO] Configuration sent")

#         # Convert audio if necessary
#         audio_bytes = convert_audio(AUDIO_FILE)
        
#         # Split audio into chunks (e.g., 4096 bytes per chunk)
#         chunk_size = 4096
#         audio_chunks = [audio_bytes[i:i+chunk_size] for i in range(0, len(audio_bytes), chunk_size)]
        
#         print(f"[INFO] Audio split into {len(audio_chunks)} chunks")
        
#         # Send audio chunks
#         send_start_time = time.time()
#         for i, chunk in enumerate(audio_chunks):
#             await websocket.send(chunk)
#             if i % 10 == 0:  # Print progress every 10 chunks
#                 print(f"[INFO] Sent chunk {i+1}/{len(audio_chunks)}")
        
#         send_end_time = time.time()
#         print(f"[INFO] All audio data sent! Time taken: {send_end_time - send_start_time:.3f} seconds")
        
#         # Listen for transcription results
#         try:
#             while True:
#                 response = await websocket.recv()
#                 try:
#                     # Try to parse as JSON
#                     result = json.loads(response)
#                     print(f"[INFO] Transcription result: {result}")
#                 except json.JSONDecodeError:
#                     # Handle binary or non-JSON responses
#                     print(f"[INFO] Received non-JSON response: {response[:100]}...")
#         except websockets.exceptions.ConnectionClosed:
#             print("[INFO] Connection closed by server")

# # Run the test
# asyncio.run(test_transcription())

#  # Make sure this file exists and is in the correct format

