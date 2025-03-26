from curses import delay_output
import os
import io
import logging
from faster_whisper import WhisperModel, BatchedInferencePipeline
import numpy as np
import wave
# import ffmpeg
import shutil
import time

from .asr_interface import ASRInterface
from datetime_utils import get_current_time_string_with_milliseconds
from audio_utils import save_audio_to_file,convert_audio
import soundfile as sf
import librosa

logger = logging.getLogger("ray.serve")
logger.setLevel(logging.INFO)


from ray import serve
from ray.serve.handle import DeploymentHandle

language_codes = {
    "afrikaans": "af",
    "amharic": "am",
    "arabic": "ar",
    "assamese": "as",
    "azerbaijani": "az",
    "bashkir": "ba",
    "belarusian": "be",
    "bulgarian": "bg",
    "bengali": "bn",
    "tibetan": "bo",
    "breton": "br",
    "bosnian": "bs",
    "catalan": "ca",
    "czech": "cs",
    "welsh": "cy",
    "danish": "da",
    "german": "de",
    "greek": "el",
    "english": "en",
    "spanish": "es",
    "estonian": "et",
    "basque": "eu",
    "persian": "fa",
    "finnish": "fi",
    "faroese": "fo",
    "french": "fr",
    "galician": "gl",
    "gujarati": "gu",
    "hausa": "ha",
    "hawaiian": "haw",
    "hebrew": "he",
    "hindi": "hi",
    "croatian": "hr",
    "haitian": "ht",
    "hungarian": "hu",
    "armenian": "hy",
    "indonesian": "id",
    "icelandic": "is",
    "italian": "it",
    "japanese": "ja",
    "javanese": "jw",
    "georgian": "ka",
    "kazakh": "kk",
    "khmer": "km",
    "kannada": "kn",
    "korean": "ko",
    "latin": "la",
    "luxembourgish": "lb",
    "lingala": "ln",
    "lao": "lo",
    "lithuanian": "lt",
    "latvian": "lv",
    "malagasy": "mg",
    "maori": "mi",
    "macedonian": "mk",
    "malayalam": "ml",
    "mongolian": "mn",
    "marathi": "mr",
    "malay": "ms",
    "maltese": "mt",
    "burmese": "my",
    "nepali": "ne",
    "dutch": "nl",
    "norwegian nynorsk": "nn",
    "norwegian": "no",
    "occitan": "oc",
    "punjabi": "pa",
    "polish": "pl",
    "pashto": "ps",
    "portuguese": "pt",
    "romanian": "ro",
    "russian": "ru",
    "sanskrit": "sa",
    "sindhi": "sd",
    "sinhalese": "si",
    "slovak": "sk",
    "slovenian": "sl",
    "shona": "sn",
    "somali": "so",
    "albanian": "sq",
    "serbian": "sr",
    "sundanese": "su",
    "swedish": "sv",
    "swahili": "sw",
    "tamil": "ta",
    "telugu": "te",
    "tajik": "tg",
    "thai": "th",
    "turkmen": "tk",
    "tagalog": "tl",
    "turkish": "tr",
    "tatar": "tt",
    "ukrainian": "uk",
    "urdu": "ur",
    "uzbek": "uz",
    "vietnamese": "vi",
    "yiddish": "yi",
    "yoruba": "yo",
    "chinese": "zh",
    "cantonese": "yue",
}

import shutil
import time

@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={"min_replicas": 1, "max_replicas": 2},
)

# @serve.deployment(
#     ray_actor_options={"num_cpus": 1},
#     autoscaling_config={"min_replicas": 1, "max_replicas": 2},
# )

# "deepdml/faster-whisper-large-v3-turbo-ct2"
# indian accent finetuned: QuantiPhy/whisper-finetuned-indian-accent-ct2
class FasterWhisperASR(ASRInterface):
    def __init__(self, **kwargs):
        model_size = kwargs.get('model_size', "deepdml/faster-whisper-large-v3-turbo-ct2")
        # Run on GPU with FP16
        logger.info(f"Using model {model_size} for transcription")
        self.asr_pipeline = WhisperModel(
            model_size, device="cuda", compute_type="float16")
        
    async def batch_transcribe(self,client):
        # file_path = await save_audio_to_file(client.scratch_buffer, client.get_file_name())
        audio_stream = io.BytesIO(client.scratch_buffer)

        language = None if client.config['language'] is None else language_codes.get(
            client.config['language'].lower())
        
        self.batched_model = BatchedInferencePipeline(model=self.asr_pipeline)

        segments, info = self.batched_model.transcribe(
            audio_stream, word_timestamps=True, language="en",batch_size=16,beam_size=2)

        # segments = list(segments)  # The transcription will actually run here.
        # os.remove(file_path)

        flattened_words = [
            word for segment in segments for word in segment.words]

        to_return = {
            "language": info.language,
            "language_probability": info.language_probability,
            "text": ' '.join([s.text.strip() for s in segments]),
            "words":
            [
                {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability} for w in flattened_words
            ]
        }
        return to_return

        
    async def transcribe(self, client):
        file_path = await save_audio_to_file(client.scratch_buffer, client.get_file_name())

        language = None if client.config['language'] is None else language_codes.get(
            client.config['language'].lower())
        segments, info = self.asr_pipeline.transcribe(
            file_path, word_timestamps=True, language="en", beam_size=5, condition_on_previous_text=False)

        segments = list(segments)  # The transcription will actually run here.
        
        # Save a copy for debugging
        current_milli_time = int(round(time.time() * 1000))
        debug_dir = "debugging"
        os.makedirs(debug_dir, exist_ok=True)
        base_name, ext = os.path.splitext(os.path.basename(file_path))
        debug_file_path = os.path.join(debug_dir, f"{base_name}_{current_milli_time}{ext}")
        shutil.copy2(file_path, debug_file_path)
        
        # Note: File is no longer deleted
        # if os.path.exists(file_path):
        #     os.remove(file_path)

        flattened_words = [
            word for segment in segments for word in segment.words]

        to_return = {
            "language": info.language,
            "language_probability": info.language_probability,
            "text": ' '.join([s.text.strip() for s in segments]),
            "words":
            [
                {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability} for w in flattened_words
            ]
        }
        return to_return



    async def transcribe(self, client, debug_output):
        file_path = await save_audio_to_file(client.scratch_buffer, client.get_file_name())

        language = None if client.config['language'] is None else language_codes.get(
            client.config['language'].lower())

         # Save a copy for debugging
        current_milli_time = int(round(time.time() * 1000))
        debug_dir = "debugging"
        os.makedirs(debug_dir, exist_ok=True)
        base_name, ext = os.path.splitext(os.path.basename(file_path))
        debug_file_path = os.path.join(debug_dir, f"{base_name}_{current_milli_time}{ext}")
        shutil.copy2(file_path, debug_file_path)
    
        current_index = len(debug_output["transcriptions_timestamp"])
        debug_output["transcriptions_timestamp"].append({"transcription_index": current_index, "start_time": get_current_time_string_with_milliseconds(), "end_time": None, "duration": None})
        segments, info = self.asr_pipeline.transcribe(
            file_path, word_timestamps=True, language="en",beam_size=2)
        
        debug_output["transcriptions_timestamp"][current_index]["end_time"] = get_current_time_string_with_milliseconds()

        segments = list(segments)  # The transcription will actually run here.
        
        if os.path.exists(file_path):
            os.remove(file_path)

        flattened_words = [
            word for segment in segments for word in segment.words]

        to_return = {
            "language": info.language,
            "language_probability": info.language_probability,
            "text": ' '.join([s.text.strip() for s in segments]),
            "words":
            [
                {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability} for w in flattened_words
            ],
            "debug_output": debug_output
        }
        return to_return
    
    # async def transcribe_byte_stream(self, client):
    #     audio_stream = io.BytesIO(client.scratch_buffer)

    #     language = None if client.config['language'] is None else language_codes.get(
    #         client.config['language'].lower())
    #     segments, info = self.asr_pipeline.transcribe(
    #         audio_stream, word_timestamps=True, language="en",beam_size=5)

    #     segments = list(segments)  # The transcription will actually run here.
        
    #     # if os.path.exists(file_path):
    #     #     os.remove(file_path)

    #     flattened_words = [
    #         word for segment in segments for word in segment.words]

    #     to_return = {
    #         "language": info.language,
    #         "language_probability": info.language_probability,
    #         "text": ' '.join([s.text.strip() for s in segments]),
    #         "words":
    #         [
    #             {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability} for w in flattened_words
    #         ]
    #     }
    #     return to_return
    
    


    async def transcribe_byte_stream(self, client, debug_output):
        try:
            # Detailed buffer logging
            logger.info(f"Total buffer size: {len(client.scratch_buffer)} bytes")
            logger.info(f"Buffer type: {type(client.scratch_buffer)}")

            # Create BytesIO stream
            audio_data=await convert_audio(client.scratch_buffer)
            audio_stream = io.BytesIO(audio_data)

            # Additional audio format debugging
            try:
                
                # Read audio data
                audio_data, samplerate = sf.read(audio_stream)
                
                # Log original audio characteristics
                logger.info(f"Original Sample Rate: {samplerate} Hz")
                logger.info(f"Original Audio Shape: {audio_data.shape}")
                logger.info(f"Original Audio Type: {audio_data.dtype}")

                # Check and convert audio format if needed
                audio_modified = False
                
                # 1. Check and convert sample rate to 16kHz
                if samplerate != 16000:
                    logger.info(f"Converting sample rate from {samplerate} to 16000 Hz")
                    audio_data = librosa.resample(audio_data, orig_sr=samplerate, target_sr=16000)
                    audio_modified = True

                # 2. Check and convert to mono if multi-channel
                if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                    logger.info(f"Converting from {audio_data.shape[1]} channels to mono")
                    audio_data = audio_data.mean(axis=1)
                    audio_modified = True

                # 3. Check and convert to float64 if needed
                if audio_data.dtype != np.float64:
                    logger.info(f"Converting audio type from {audio_data.dtype} to float64")
                    audio_data = audio_data.astype(np.float64)
                    audio_modified = True

                # If any modifications were made, create new audio stream
                if audio_modified:
                    logger.info("Audio was modified to match required format")
                    audio_stream = io.BytesIO()
                    sf.write(audio_stream, audio_data, 16000, format='RAW')
                    audio_stream.seek(0)
                else:
                    logger.info("Audio already in correct format, using original")
                  # audio_stream = io.BytesIO(client.scratch_buffer)

                # Log final audio characteristics
                if audio_modified:
                    logger.info("Final audio format:")
                    logger.info(f"Sample Rate: 16000 Hz")
                    logger.info(f"Audio Shape: {audio_data.shape}")
                    logger.info(f"Audio Type: {audio_data.dtype}")
                
            except Exception as format_error:
                logger.error(f"Audio format processing error: {format_error}")
                return {
                    "error": str(format_error),
                    "buffer_details": {
                        "size": len(client.scratch_buffer),
                        "type": str(type(client.scratch_buffer))
                    }
                }

            # Language detection
            language = None if client.config['language'] is None else language_codes.get(
                client.config['language'].lower())
            
            audio_stream = io.BytesIO(audio_data)


            # Transcription
            logger.info("Starting transcription with Whisper model")
            segments, info = self.asr_pipeline.transcribe(
                audio_stream, 
                word_timestamps=True, 
                language="en", 
                beam_size=5
            )
            logger.info("Completed transcription with Whisper model")

            segments = list(segments)  # Ensure segments is a list
            
            # Flatten words
            flattened_words = [
                word for segment in segments for word in segment.words
            ]

            # Prepare return dictionary
            to_return = {
                "language": info.language,
                "language_probability": info.language_probability,
                "text": ' '.join([s.text.strip() for s in segments]),
                "words": [
                    {
                        "word": w.word, 
                        "start": w.start, 
                        "end": w.end, 
                        "probability": w.probability
                    } for w in flattened_words
                ],
                "debug_info": {
                    "original_buffer_size": len(client.scratch_buffer),
                    "processed_audio_format": {
                        "sample_rate": 16000 if audio_modified else samplerate,
                        "channels": 1 if audio_modified else (audio_data.shape[1] if len(audio_data.shape) > 1 else 1),
                        "data_type": str(audio_data.dtype)
                    }
                }
            }

            logger.info(f"Successfully processed transcription with {len(flattened_words)} words")
            return to_return

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return {
                "error": str(e),
                "buffer_details": {
                    "size": len(client.scratch_buffer),
                    "type": str(type(client.scratch_buffer))
                }
            }

    async def transcribe_raw(self, client, debug_output):
        """
        Transcribe raw audio bytes directly without saving to a WAV file.
        Expects 16-bit PCM audio data at 16kHz sample rate.
        """
        try:
            # Log input buffer details
            logger.info(f"Received audio buffer - Size: {len(client.scratch_buffer)} bytes")
            logger.info(f"Buffer type: {type(client.scratch_buffer)}")
            
            # Convert raw PCM bytes to numpy array
            audio_data = np.frombuffer(client.scratch_buffer, dtype=np.int16)
            logger.info(f"Converted to numpy array with shape: {audio_data.shape}")
            
            # Normalize to float32 in range [-1, 1]
            audio_data = audio_data.astype(np.float32) / 32768.0
            logger.info(f"Normalized audio data to float32")
            
            # Language detection
            language = None if client.config['language'] is None else language_codes.get(
                client.config['language'].lower())
            logger.info(f"Using language: {language if language else 'auto-detect'}")
            
            # Transcription
            logger.info("Starting transcription with Whisper model")
            start_time = time.time()
            
            current_index = len(debug_output["transcriptions_timestamp"])
            debug_output["transcriptions_timestamp"].append({"transcription_index": current_index, "start_time": get_current_time_string_with_milliseconds(), "end_time": None, "duration": None})
            segments, info = self.asr_pipeline.transcribe(
                audio_data, 
                word_timestamps=True, 
                language="en", 
                beam_size=5,
                vad_filter=True,
            )
            debug_output["transcriptions_timestamp"][current_index]["end_time"] = get_current_time_string_with_milliseconds()
            
            transcription_time = time.time() - start_time
            logger.info(f"Completed transcription in {transcription_time:.2f} seconds")
            logger.info(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

            segments = list(segments)  # Ensure segments is a list
            logger.info(f"Generated {len(segments)} segments")
            
            # Flatten words
            flattened_words = [
                word for segment in segments for word in segment.words
            ]
            logger.info(f"Extracted {len(flattened_words)} words")

            # Prepare return dictionary
            to_return = {
                "language": info.language,
                "language_probability": info.language_probability,
                "text": ' '.join([s.text.strip() for s in segments]),
                "words": [
                    {
                        "word": w.word, 
                        "start": w.start, 
                        "end": w.end, 
                        "probability": w.probability
                    } for w in flattened_words
                ],
                "debug_info": {
                    "original_buffer_size": len(client.scratch_buffer),
                    "audio_shape": audio_data.shape,
                    "audio_dtype": str(audio_data.dtype),
                    "audio_duration": f"{len(audio_data) / 16000:.2f} seconds"  # Assuming 16kHz sample rate
                },
                "debug_output": debug_output
            }

            logger.info(f"Successfully processed transcription with {len(flattened_words)} words")
            logger.info(f"Transcribed text: {to_return['text'][:100]}...")  # Log first 100 chars of text
            return to_return

        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "error": str(e),
                "buffer_details": {
                    "size": len(client.scratch_buffer),
                    "type": str(type(client.scratch_buffer))
                }
            }


