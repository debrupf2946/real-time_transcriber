import wave
import os
import io
async def save_audio_to_file(audio_data, file_name, audio_dir="audio_files", audio_format="wav"):
    """
    Saves the audio data to a file.

    :param client_id: Unique identifier for the client.
    :param audio_data: The audio data to save.
    :param file_counters: Dictionary to keep track of file counts for each client.
    :param audio_dir: Directory where audio files will be saved.
    :param audio_format: Format of the audio file.
    :return: Path to the saved audio file.
    """

    os.makedirs(audio_dir, exist_ok=True)
    
    file_path = os.path.join(audio_dir, file_name)

    with wave.open(file_path, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Assuming mono audio
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(audio_data)

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Failed to create audio file at {file_path}")


    return file_path


async def convert_audio(audio_data):
    """
    Ensures the audio is in the correct format (16kHz, 16-bit PCM, mono).
    Appends 0.6 seconds of silence at the end.
    Returns the correctly formatted audio as bytes.

    Args:
        audio_data: Raw audio data in bytes (expected to be 16-bit PCM)

    Returns:
        bytes: Audio data in WAV format with correct specifications
    """
    # Calculate silence duration (0.6 seconds at 16kHz, 16-bit)
    silence_duration = int(0.6 * 16000 * 2)  # 0.6s * 16kHz * 2 bytes per sample
    silence = b'\x00' * silence_duration

    # Write to an in-memory WAV buffer
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(16000)  # 16kHz
        wf.writeframes(audio_data)
        wf.writeframes(silence)  # Append silence

    return buffer.getvalue()
