import os
import asyncio
import json
import time
from fastapi import WebSocket

from .buffering_strategy_interface import BufferingStrategyInterface
from ray.serve.handle import DeploymentHandle

import logging
logger = logging.getLogger("ray.serve")
logger.setLevel(logging.DEBUG)

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


class SilenceAtEndOfChunk(BufferingStrategyInterface):
    """
    A buffering strategy that processes audio only when silence is detected at the end of the buffer.
    """
    def __init__(self, client, **kwargs):
        logger.info("Initializing SilenceAtEndOfChunk buffering strategy")
        self.client = client
        
        self.silence_threshold_seconds = float(os.environ.get('BUFFERING_SILENCE_THRESHOLD_SECONDS', kwargs.get('silence_threshold_seconds', 0.5)))
        self.min_speech_seconds = float(os.environ.get('BUFFERING_MIN_SPEECH_SECONDS', kwargs.get('min_speech_seconds', 0.5)))
        self.max_buffer_seconds = float(os.environ.get('BUFFERING_MAX_BUFFER_SECONDS', kwargs.get('max_buffer_seconds', 3000.0)))
        self.error_if_not_realtime = kwargs.get('error_if_not_realtime', False)
        
        self.processing_flag = False
        self.last_chunk_time = time.time()
        
        logger.debug(f"Silence threshold: {self.silence_threshold_seconds}s, Min speech: {self.min_speech_seconds}s, Max buffer: {self.max_buffer_seconds}s")

    def process_audio(self, websocket: WebSocket, vad_handle, asr_handle):
        logger.info("Starting audio processing")
        self.last_chunk_time = time.time()
        
        if self.processing_flag:
            logger.debug("Skipping processing: already in progress")
            return

        max_buffer_size_bytes = self.max_buffer_seconds * self.client.sampling_rate * self.client.samples_width
        if len(self.client.buffer) > max_buffer_size_bytes:
            logger.warning("Buffer exceeded maximum size, triggering forced processing")
            self.processing_flag = True
            asyncio.create_task(self.process_audio_async(websocket, vad_handle, asr_handle))
            return

        min_buffer_size = self.silence_threshold_seconds * self.client.sampling_rate * self.client.samples_width
        if len(self.client.buffer) < min_buffer_size:
            logger.debug("Insufficient buffer size, waiting for more data")
            return

        logger.info("Checking for silence")
        asyncio.create_task(self.check_silence_and_process(websocket, vad_handle, asr_handle))

    async def check_silence_and_process(self, websocket: WebSocket, vad_handle, asr_handle):
        logger.info("Entered check_silence_and_process function")
        if self.processing_flag:
            logger.debug("Skipping silence check: already processing")
            return

        original_scratch_buffer = self.client.scratch_buffer
        self.client.scratch_buffer = bytearray(self.client.buffer)

        vad_results = await vad_handle.detect_activity.remote(client=self.client)
        logger.debug(f"VAD results: {vad_results}")

        if not vad_results:
            logger.info("No speech detected, restoring buffer")
            self.client.scratch_buffer = original_scratch_buffer
            return

        buffer_duration = len(self.client.buffer) / (self.client.sampling_rate * self.client.samples_width)
        logger.debug(f"Buffer duration: {buffer_duration:.2f}s")

        if vad_results[-1]['end'] < (buffer_duration - self.silence_threshold_seconds):
            total_speech_duration = sum(segment['end'] - segment['start'] for segment in vad_results)
            if total_speech_duration >= self.min_speech_seconds:
                logger.info(f"Silence detected, processing {buffer_duration:.2f}s of audio")
                self.processing_flag = True
                self.client.buffer.clear()
                await self.process_audio_async(websocket, vad_handle, asr_handle)
            else:
                logger.info("Not enough speech detected, restoring buffer")
                self.client.scratch_buffer = original_scratch_buffer
        else:
            logger.info("No silence detected at buffer end, restoring buffer")
            self.client.scratch_buffer = original_scratch_buffer

    async def process_audio_async(self, websocket: WebSocket, vad_handle, asr_handle: DeploymentHandle):
        try:
            start = time.time()
            logger.info("Starting asynchronous audio processing")
            client_config_language = self.client.config.get("language")
            language = "en" if client_config_language is None else language_codes.get(
                client_config_language.lower())
            
            transcription = await asr_handle.transcribe.remote(language, bytearray(self.client.scratch_buffer))
            self.client.increment_file_counter()

            if transcription['text'].strip():
                end = time.time()
                transcription['processing_time'] = end - start
                json_transcription = json.dumps(transcription)
                await websocket.send_text(json_transcription)
                logger.info(f"Sent transcription: {json_transcription}")
            
            logger.debug(f"Processed {len(self.client.scratch_buffer) / (self.client.sampling_rate * self.client.samples_width):.2f}s of audio")
            self.client.scratch_buffer.clear()
        except Exception as e:
            logger.error(f"Error during async processing: {e}")
        finally:
            logger.info("Finished async processing, resetting flag")
            self.processing_flag = False
