import asyncio
import threading
import queue
import pandas as pd
import time
import wave
import os
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor
from my_test import whisper_transcription
from azure_recorded import azure_transcribe

def bytes_to_wav(audio_bytes, filename):
    """Convert audio bytes to WAV file dynamically."""
    filepath = f"temp_{filename}.wav"
    with wave.open(filepath, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(audio_bytes)
    return filepath

# def load_and_stream_samples(num_samples=1000, batch_size=50):
#     print("[LOADER] Loading dataset samples")
#     ds = load_dataset('speechbrain/LargeScaleASR', name="small", streaming=True, trust_remote_code=True)
#     train_iter = iter(ds["train"])
    
#     batch_counter = 0
#     while True:
#         azure_queue = queue.Queue(maxsize=batch_size)
#         whisper_queue = queue.Queue(maxsize=batch_size)
#         samples_loaded = 0

#         for _ in range(batch_size):
#             try:
#                 sample = next(train_iter)
#                 audio_bytes = sample["wav"]["bytes"]
#                 azure_queue.put((batch_counter * batch_size + samples_loaded, audio_bytes))
#                 whisper_queue.put((batch_counter * batch_size + samples_loaded, audio_bytes))
#                 samples_loaded += 1
#             except StopIteration:
#                 print("[LOADER] No more samples available")
#                 break

#         if samples_loaded == 0:
#             break  # No more data to load

#         yield azure_queue, whisper_queue, batch_counter
#         batch_counter += 1


def load_and_stream_samples(num_samples=1000, batch_size=50):
    print(f"[LOADER] Loading dataset with {num_samples} samples in batches of {batch_size}")
    ds = load_dataset('speechbrain/LargeScaleASR', name="small", streaming=True, trust_remote_code=True)
    train_iter = iter(ds["train"])
    
    total_samples_processed = 0
    batch_counter = 0
    
    while total_samples_processed < num_samples:
        azure_queue = queue.Queue(maxsize=batch_size)
        whisper_queue = queue.Queue(maxsize=batch_size)
        samples_in_current_batch = 0
        
        # Calculate how many samples we need in this batch
        samples_to_load = min(batch_size, num_samples - total_samples_processed)
        
        for _ in range(samples_to_load):
            try:
                sample = next(train_iter)
                audio_bytes = sample["wav"]["bytes"]
                azure_queue.put((total_samples_processed, audio_bytes))
                whisper_queue.put((total_samples_processed, audio_bytes))
                samples_in_current_batch += 1
                total_samples_processed += 1
            except StopIteration:
                print("[LOADER] No more samples available in dataset")
                break
        
        if samples_in_current_batch == 0:
            break  # No more data to load
        
        print(f"[LOADER] Yielding batch {batch_counter} with {samples_in_current_batch} samples. Total processed: {total_samples_processed}/{num_samples}")
        yield azure_queue, whisper_queue, batch_counter
        batch_counter += 1
        
        # If we've processed all requested samples, stop
        if total_samples_processed >= num_samples:
            print(f"[LOADER] Reached requested sample count of {num_samples}")
            break

def process_azure_transcription(azure_queue, results):
    while not azure_queue.empty():
        sample_id, audio_input = azure_queue.get()
        if isinstance(audio_input, bytes):
            wav_path = bytes_to_wav(audio_input, f"azure_{sample_id}")
        else:
            wav_path = audio_input  # Already a file path
        text, latency = azure_transcribe(wav_path)
        results["azure_results"][sample_id] = {"text": text, "latency": latency}
        if isinstance(audio_input, bytes):
            os.remove(wav_path)
        print(f"[AZURE] Processed Sample {sample_id}")

def process_whisper_transcription(whisper_queue, results):
    while not whisper_queue.empty():
        sample_id, audio_input = whisper_queue.get()
        if isinstance(audio_input, bytes):
            wav_path = bytes_to_wav(audio_input, f"whisper_{sample_id}")
        else:
            wav_path = audio_input  # Already a file path
        text, latency = asyncio.run(whisper_transcription(wav_path))
        results["whisper_results"][sample_id] = {"text": text, "latency": latency}
        if isinstance(audio_input, bytes):
            os.remove(wav_path)
        print(f"[WHISPER] Processed Sample {sample_id}")

def save_results_to_dataframe(results, batch_id, folder="results"):
    """Save results every batch to a CSV file."""
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"transcriptions_batch_{batch_id}.csv")
    print(f"[SAVE] Saving batch {batch_id} to {filename}")

    data = []
    for sample_id in results["azure_results"]:
        data.append([
            f"Sample_{sample_id}",
            results["azure_results"].get(sample_id, {}).get("text", ""),
            results["azure_results"].get(sample_id, {}).get("latency", None),
            results["whisper_results"].get(sample_id, {}).get("text", ""),
            results["whisper_results"].get(sample_id, {}).get("latency", None)
        ])
    
    columns = [
        ("Sample Name", ""),
        ("Azure Transcription", "Transcribed Text"),
        ("Azure Transcription", "Latency"),
        ("Whisper Transcription", "Transcribed Text"),
        ("Whisper Transcription", "Latency")
    ]
    
    df = pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(columns))
    df.to_csv(filename, index=False)
    print(f"[SAVE] Batch {batch_id} results saved to {filename}")

def merge_csv_files(output_filename="final_transcriptions.csv", folder="results"):
    """Merge all batch CSV files into one final file."""
    print("[MERGE] Merging all batch results into one file")
    files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.startswith("transcriptions_batch_")])
    
    if not files:
        print("[MERGE] No batch files found!")
        return

    dfs = [pd.read_csv(f, header=[0,1]) for f in files]
    final_df = pd.concat(dfs, ignore_index=True)
    final_df.to_csv(output_filename, index=False)
    print(f"[MERGE] Merged results saved to {output_filename}")

def main():
    batch_size = 50
    results_folder = "results"

    for azure_queue, whisper_queue, batch_id in load_and_stream_samples(num_samples=1000, batch_size=batch_size):
        results = {"azure_results": {}, "whisper_results": {}}
        
        azure_thread = threading.Thread(target=process_azure_transcription, args=(azure_queue, results))
        whisper_thread = threading.Thread(target=process_whisper_transcription, args=(whisper_queue, results))
        
        azure_thread.start()
        whisper_thread.start()
        
        azure_thread.join()
        whisper_thread.join()
        
        save_results_to_dataframe(results, batch_id, folder=results_folder)

    merge_csv_files(output_filename="final_transcriptions.csv", folder=results_folder)

if __name__ == "__main__":
    main()




# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# import asyncio
# import threading
# import queue
# import pandas as pd
# import time
# import wave
# import os
# from datasets import load_dataset
# from concurrent.futures import ThreadPoolExecutor
# from my_test import whisper_transcription
# from azure_recorded import azure_transcribe

# def bytes_to_wav(audio_bytes, filename):
#     """Convert audio bytes to WAV file dynamically."""
#     filepath = f"temp_{filename}.wav"
#     with wave.open(filepath, 'wb') as wav_file:
#         wav_file.setnchannels(1)
#         wav_file.setsampwidth(2)
#         wav_file.setframerate(16000)
#         wav_file.writeframes(audio_bytes)
#     return filepath

# def load_and_stream_samples(num_samples=10):
#     print("[LOADER] Loading dataset samples")
#     ds = load_dataset('speechbrain/LargeScaleASR', name="clean", streaming=True, trust_remote_code=True)
#     train_iter = iter(ds["train"])
#     azure_queue = queue.Queue(maxsize=num_samples)
#     whisper_queue = queue.Queue(maxsize=num_samples)
    
#     for i in range(num_samples):
#         try:
#             sample = next(train_iter)
#             audio_bytes = sample["wav"]["bytes"]
#             azure_queue.put((i, audio_bytes))
#             whisper_queue.put((i, audio_bytes))
#             print(f"[LOADER] Sample {i} loaded and added to queues")
#         except StopIteration:
#             print("[LOADER] No more samples available")
#             break
    
#     return azure_queue, whisper_queue

# def load_audio_from_directory(directory="audio_chunks"):
#     print(f"[LOADER] Loading audio files from {directory}")
#     azure_queue = queue.Queue()
#     whisper_queue = queue.Queue()
    
#     audio_files = [f for f in os.listdir(directory) if f.endswith(".wav")]
#     for i, filename in enumerate(audio_files[:10]):
#         filepath = os.path.join(directory, filename)
#         azure_queue.put((i, filepath))
#         whisper_queue.put((i, filepath))
#         print(f"[LOADER] {filename} loaded and added to queues")
    
#     return azure_queue, whisper_queue

# def process_azure_transcription(azure_queue, results):
#     while not azure_queue.empty():
#         sample_id, audio_input = azure_queue.get()
#         if isinstance(audio_input, bytes):
#             wav_path = bytes_to_wav(audio_input, f"azure_{sample_id}")
#         else:
#             wav_path = audio_input  # Already a file path
#         text, latency = azure_transcribe(wav_path)
#         results["azure_results"][sample_id] = {"text": text, "latency": latency}
#         if isinstance(audio_input, bytes):
#             os.remove(wav_path)
#         print(f"[AZURE] Processed Sample {sample_id}")

# def process_whisper_transcription(whisper_queue, results):
#     while not whisper_queue.empty():
#         sample_id, audio_input = whisper_queue.get()
#         if isinstance(audio_input, bytes):
#             wav_path = bytes_to_wav(audio_input, f"whisper_{sample_id}")
#         else:
#             wav_path = audio_input  # Already a file path
#         text, latency = asyncio.run(whisper_transcription(wav_path))
#         results["whisper_results"][sample_id] = {"text": text, "latency": latency}
#         if isinstance(audio_input, bytes):
#             os.remove(wav_path)
#         print(f"[WHISPER] Processed Sample {sample_id}")

# # def save_results_to_dataframe(results, filename="transcriptions.csv"):
# #     print(f"[SAVE] Saving results to {filename}")
# #     data = []
# #     for sample_id in results["azure_results"]:
# #         data.append({
# #             "audio_sample_name": f"Sample_{sample_id}",
# #             "Azure_transcription": results["azure_results"].get(sample_id, {"text": "", "latency": None}),
# #             "Whisper_transcription": results["whisper_results"].get(sample_id, {"text": "", "latency": None})
# #         })
    
# #     df = pd.DataFrame(data)
# #     df.to_csv(filename, index=False)
# #     print(f"[SAVE] Results saved to {filename}")
# #     return df
# def save_results_to_dataframe(results, filename="transcriptions.csv"):
#     print(f"[SAVE] Saving results to {filename}")
    
#     data = []
#     for sample_id in results["azure_results"]:
#         data.append([
#             f"Sample_{sample_id}",
#             results["azure_results"].get(sample_id, {}).get("text", ""),
#             results["azure_results"].get(sample_id, {}).get("latency", None),
#             results["whisper_results"].get(sample_id, {}).get("text", ""),
#             results["whisper_results"].get(sample_id, {}).get("latency", None)
#         ])
    
#     # Define multi-index column names
#     columns = [
#         ("Sample Name", ""),
#         ("Azure Transcription", "Transcribed Text"),
#         ("Azure Transcription", "Latency"),
#         ("Whisper Transcription", "Transcribed Text"),
#         ("Whisper Transcription", "Latency")
#     ]
    
#     # Create a MultiIndex DataFrame
#     df = pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(columns))

#     df.to_csv(filename, index=False)
#     print(f"[SAVE] Results saved to {filename}")
    
#     return df

# def main(streaming=True):
#     results = {"azure_results": {}, "whisper_results": {}}
    
#     if streaming:
#         num_samples = 1000 # Adjust as needed
#         azure_queue, whisper_queue = load_and_stream_samples(num_samples)
#     else:
#         azure_queue, whisper_queue = load_audio_from_directory()
    
#     azure_thread = threading.Thread(target=process_azure_transcription, args=(azure_queue, results))
#     whisper_thread = threading.Thread(target=process_whisper_transcription, args=(whisper_queue, results))
    
#     azure_thread.start()
#     whisper_thread.start()
    
#     azure_thread.join()
#     whisper_thread.join()
    
#     save_results_to_dataframe(results)

# if __name__ == "__main__":
#     main(streaming=True)  # Change to True to use dataset streaming

# import asyncio
# import queue
# import wave
# import os
# from datasets import load_dataset
# from my_test import whisper_transcription

# def bytes_to_wav(audio_bytes, filename):
#     """Convert audio bytes to WAV file dynamically."""
#     filepath = f"temp_{filename}.wav"
#     with wave.open(filepath, 'wb') as wav_file:
#         wav_file.setnchannels(1)
#         wav_file.setsampwidth(2)  # 16-bit PCM
#         wav_file.setframerate(16000)  # 16kHz
#         wav_file.writeframes(audio_bytes)
#     return filepath

# def load_whisper_samples(num_samples=1):
#     """Load audio samples into a queue for Whisper transcription."""
#     print("[LOADER] Loading dataset samples")
#     ds = load_dataset('speechbrain/LargeScaleASR', name="clean", streaming=True, trust_remote_code=True)
#     train_iter = iter(ds["train"])
#     whisper_queue = queue.Queue(maxsize=num_samples)
    
#     for i in range(num_samples):
#         try:
#             sample = next(train_iter)
#             audio_bytes = sample["wav"]["bytes"]
#             whisper_queue.put((i, audio_bytes))
#             print(f"[LOADER] Sample {i} loaded and added to queue")
#         except StopIteration:
#             print("[LOADER] No more samples available")
#             break
    
#     return whisper_queue

# def load_audio_from_directory(directory="audio_chunks"):
#     """Load audio files from a local directory."""
#     print(f"[LOADER] Loading audio files from {directory}")
#     whisper_queue = queue.Queue()
    
#     audio_files = [f for f in os.listdir(directory) if f.endswith(".wav")]
#     for i, filename in enumerate(audio_files[:2]):
#         filepath = os.path.join(directory, filename)
#         whisper_queue.put((i, filepath))
#         print(f"[LOADER] {filename} loaded and added to queue")
    
#     return whisper_queue

# def process_whisper_transcription(whisper_queue, results):
#     """Process the queue of audio samples and run Whisper transcription."""
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)

#     while not whisper_queue.empty():
#         sample_id, audio_input = whisper_queue.get()
        
#         if isinstance(audio_input, bytes):
#             wav_path = bytes_to_wav(audio_input, f"whisper_{sample_id}")
#         else:
#             wav_path = audio_input  # Already a file path
        
#         try:
#             print(f"[WHISPER] Processing Sample {sample_id}")
#             text, latency = loop.run_until_complete(whisper_transcription(wav_path))
#             results["whisper_results"][sample_id] = {"text": text, "latency": latency}
#         except Exception as e:
#             print(f"[ERROR] Whisper transcription failed for Sample {sample_id}: {e}")
#         finally:
#             if isinstance(audio_input, bytes):
#                 os.remove(wav_path)
    
#     loop.close()

# def main(streaming=True):
#     results = {"whisper_results": {}}
    
#     if streaming:
#         whisper_queue = load_whisper_samples(num_samples=1)  # Adjust as needed
#     else:
#         whisper_queue = load_audio_from_directory()
    
#     process_whisper_transcription(whisper_queue, results)
#     print("[DEBUG] Whisper Results:", results)

# if __name__ == "__main__":
#     main(streaming=False)  # Change to True to use dataset streaming
