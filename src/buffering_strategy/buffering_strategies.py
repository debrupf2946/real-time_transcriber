import asyncio
import json
import logging
import os
import time

from fastapi import WebSocket
from multiprocessing.reduction import send_handle
from ray.serve.handle import DeploymentHandle

from .buffering_strategy_interface import BufferingStrategyInterface

logger = logging.getLogger("ray.serve")
logger.setLevel(logging.DEBUG)


class SilenceAtEndOfChunk(BufferingStrategyInterface):
    """A buffering strategy that processes audio only when silence is detected at the end of the buffer."""

    def __init__(self, client, **kwargs):
        logger.info("Initializing SilenceAtEndOfChunk buffering strategy")
        self.client = client

        self.silence_threshold_seconds = float(os.environ.get('BUFFERING_SILENCE_THRESHOLD_SECONDS', kwargs.get('silence_threshold_seconds', 0.5)))
        self.min_speech_seconds = float(os.environ.get('BUFFERING_MIN_SPEECH_SECONDS', kwargs.get('min_speech_seconds', 0.5)))
        self.max_buffer_seconds = float(os.environ.get('BUFFERING_MAX_BUFFER_SECONDS', kwargs.get('max_buffer_seconds', 3000.0)))
        self.error_if_not_realtime = kwargs.get('error_if_not_realtime', False)
        self.buffer_context_seconds_for_vad = float(os.environ.get('BUFFERING_BUFFER_CONTEXT_SECONDS_FOR_VAD', 5.0))

        self.processing_flag = False
        self.last_chunk_time = time.time()

        logger.debug(f"Silence threshold: {self.silence_threshold_seconds}s, Min speech: {self.min_speech_seconds}s, Max buffer: {self.max_buffer_seconds}s")

    def process_audio(self, websocket: WebSocket, vad_handle, asr_handle, debug_output):
        logger.debug("Starting audio processing")
        self.last_chunk_time = time.time()

        if self.processing_flag:
            logger.debug("Skipping processing: already in progress")
            return

        if not self.client:
            logger.error("Client is None, cannot process audio")
            return
        # Ensure sampling_rate and samples_width have valid values
        sampling_rate = self.client.sampling_rate if getattr(self.client, 'sampling_rate', None) is not None else 16000
        samples_width = self.client.samples_width if getattr(self.client, 'samples_width', None) is not None else 2

        max_buffer_size_bytes = self.max_buffer_seconds * sampling_rate * samples_width
        if len(self.client.buffer) > max_buffer_size_bytes:
            logger.warning("buffer exceeded maximum size, triggering forced processing")
            self.processing_flag = True
            asyncio.create_task(self.process_audio_async(websocket, vad_handle, asr_handle, debug_output))
            return

        min_buffer_size = self.silence_threshold_seconds * sampling_rate * samples_width
        if len(self.client.buffer) < min_buffer_size:
            logger.debug("Insufficient buffer size, waiting for more data")
            return

        logger.debug("Checking for silence")
        asyncio.create_task(self.check_silence_and_process(websocket, vad_handle, asr_handle, debug_output))

    async def check_silence_and_process(self, websocket: WebSocket, vad_handle, asr_handle, debug_output):

        if self.processing_flag:
            logger.debug("Skipping silence check: already processing")
            return

        if not self.client:
            logger.error("Client is None, cannot check silence")
            return

        # Safely get the original scratch buffer, or default to an empty bytearray
        original_scratch_buffer = getattr(self.client, 'scratch_buffer', bytearray())
        try:
            bytes_per_second = self.client.sampling_rate * self.client.samples_width
            bytes_in_buffer_context_seconds = int(bytes_per_second * self.buffer_context_seconds_for_vad)
            if (len(self.client.buffer) > bytes_in_buffer_context_seconds):
                self.client.scratch_buffer = bytearray(self.client.buffer[-bytes_in_buffer_context_seconds:])
            else:
                self.client.scratch_buffer = bytearray(self.client.buffer)
        except Exception as e:
            logger.error(f"Error creating scratch buffer: {e}")
            return

        vad_results = await vad_handle.detect_activity.remote(client=self.client, debug_output=debug_output)
        logger.debug(f"VAD results: {vad_results}")

        if not vad_results:
            logger.info("No speech detected, restoring buffer")
            self.client.scratch_buffer = original_scratch_buffer
            return

        buffer_length = len(self.client.buffer) if self.client.buffer else 0
        bytes_per_second = self.client.sampling_rate * self.client.samples_width
        buffer_duration = buffer_length / bytes_per_second if bytes_per_second else 0
        logger.debug(f"Buffer duration: {buffer_duration}")

        # Check if vad_results has the expected structure
        if not isinstance(vad_results, list) or 'end' not in vad_results[-1]:
            logger.error("Invalid VAD results structure")
            self.client.scratch_buffer = original_scratch_buffer
            return

        if vad_results[-1]['end'] < (buffer_duration - self.silence_threshold_seconds):
            total_speech_duration = sum(segment.get('end', 0) - segment.get('start', 0) for segment in vad_results)
            if total_speech_duration >= self.min_speech_seconds:
                logger.debug(f"Silence detected, processing {buffer_duration:.2f}s of audio")
                self.processing_flag = True
                self.client.buffer.clear()
                await self.process_audio_async(websocket, send_handle, asr_handle, debug_output)
            else:
                logger.debug("Not enough speech detected, restoring buffer")
                self.client.scratch_buffer = original_scratch_buffer
        else:
            logger.debug("No silence detected at buffer end, restoring buffer")
            self.client.scratch_buffer = original_scratch_buffer

    async def process_audio_async(self, websocket: WebSocket, vad_handle, asr_handle: DeploymentHandle, debug_output):
        try:
            start = time.time()
            logger.info("Starting asynchronous audio processing")
            transcription = await asr_handle.transcribe_raw.remote(client=self.client, debug_output=debug_output)

            if transcription is None:
                logger.error("Transcription returned None")
                return
            self.client.increment_file_counter()

            if transcription.get('error'):
                logger.error(f"Transcription failed: {transcription['error']}")
                return

            if transcription.get('text', '').strip():
                end = time.time()
                logger.info(f"Transcription processing time: {end - start}")
                transcription['processing_time'] = end - start
                json_transcription = json.dumps(transcription)
                await websocket.send_text(json_transcription)
                logger.info(f"Sent transcription: {json_transcription}")

            # Ensure scratch_buffer and audio properties are valid
            if hasattr(self.client, 'scratch_buffer') and self.client.scratch_buffer is not None:
                total_seconds = len(self.client.scratch_buffer) / (self.client.sampling_rate * self.client.samples_width)
                logger.info(f"Processed {total_seconds:.2f}s of audio")
                self.client.scratch_buffer.clear()
            else:
                logger.error("Client scratch_buffer is None")
        except Exception as e:
            logger.error(f"Error during async processing: {e}")
        finally:
            logger.debug("Finished async processing, resetting flag")
            self.processing_flag = False