from multiprocessing.reduction import send_handle
import os
import asyncio
import json
import time
from fastapi import WebSocket
from datetime_utils import get_current_time_string_with_milliseconds

from .buffering_strategy_interface import BufferingStrategyInterface
from ray.serve.handle import DeploymentHandle

import logging
logger = logging.getLogger("ray.serve")
logger.setLevel(logging.DEBUG)


class SilenceAtEndOfChunk(BufferingStrategyInterface):
    """
    A buffering strategy that processes audio only when silence is detected at the end of the buffer.
    """
    def __init__(self, client, **kwargs):
        logger.info("Initializing SilenceAtEndOfChunk buffering strategy")
        if client is None:
            logger.error("Client is None during initialization")
        self.client = client

        self.silence_threshold_seconds = float(os.environ.get('BUFFERING_SILENCE_THRESHOLD_SECONDS', kwargs.get('silence_threshold_seconds', 0.5)))
        self.min_speech_seconds = float(os.environ.get('BUFFERING_MIN_SPEECH_SECONDS', kwargs.get('min_speech_seconds', 0.5)))
        self.max_buffer_seconds = float(os.environ.get('BUFFERING_MAX_BUFFER_SECONDS', kwargs.get('max_buffer_seconds', 3000.0)))
        self.error_if_not_realtime = kwargs.get('error_if_not_realtime', False)
        
        self.processing_flag = False
        self.last_chunk_time = time.time()
        
        logger.debug(f"Silence threshold: {self.silence_threshold_seconds}s, Min speech: {self.min_speech_seconds}s, Max buffer: {self.max_buffer_seconds}s")

    def process_audio(self, websocket: WebSocket, vad_handle, asr_handle, debug_output):
        logger.info("Starting audio processing")
        self.last_chunk_time = time.time()
        
        if self.processing_flag:
            logger.debug("Skipping processing: already in progress")
            return

        if not self.client:
            logger.error("Client is None, cannot process audio")
            return

        if not hasattr(self.client, 'buffer') or self.client.buffer is None:
            logger.error("Client buffer is None, cannot process audio")
            return

        # Ensure sampling_rate and samples_width have valid values
        sampling_rate = self.client.sampling_rate if getattr(self.client, 'sampling_rate', None) is not None else 1
        samples_width = self.client.samples_width if getattr(self.client, 'samples_width', None) is not None else 1

        max_buffer_size_bytes = self.max_buffer_seconds * sampling_rate * samples_width
        if len(self.client.buffer) > max_buffer_size_bytes:
            logger.warning("Buffer exceeded maximum size, triggering forced processing")
            self.processing_flag = True
            asyncio.create_task(self.process_audio_async(websocket, vad_handle, asr_handle, debug_output))
            return

        min_buffer_size = self.silence_threshold_seconds * sampling_rate * samples_width
        if len(self.client.buffer) < min_buffer_size:
            logger.debug("Insufficient buffer size, waiting for more data")
            return

        logger.info("Checking for silence")
        asyncio.create_task(self.check_silence_and_process(websocket, vad_handle, asr_handle, debug_output))

    async def check_silence_and_process(self, websocket: WebSocket, vad_handle, asr_handle, debug_output):
        logger.info("Entered check_silence_and_process function")
        if self.processing_flag:
            logger.debug("Skipping silence check: already processing")
            return

        if not self.client:
            logger.error("Client is None, cannot check silence")
            return

        if not hasattr(self.client, 'buffer') or self.client.buffer is None:
            logger.error("Client buffer is None, cannot check silence")
            return

        # Safely get the original scratch buffer, or default to an empty bytearray
        original_scratch_buffer = getattr(self.client, 'scratch_buffer', bytearray())
        try:
            # Pass only the last portion of the buffer for silence detection
            
            # self.client.scratch_buffer = bytearray(self.client.buffer)
            logger.info(f"Buffer length: {len(self.client.buffer)}")
            bytes_per_second = self.client.sampling_rate * self.client.samples_width
            bytes_in_5_seconds = bytes_per_second * 5
            self.client.scratch_buffer = bytearray(self.client.buffer[-bytes_in_5_seconds:]) if len(self.client.buffer) > bytes_in_5_seconds else bytearray(self.client.buffer)
            logger.info(f"Scratch buffer length: {len(self.client.scratch_buffer)}")
        except Exception as e:
            logger.error(f"Error creating scratch buffer: {e}")
            return

        vad_results = await vad_handle.detect_activity.remote(client=self.client, debug_output=debug_output)
        logger.info(f"VAD results: {vad_results}")

        if not vad_results:
            logger.info("No speech detected, restoring buffer")
            self.client.scratch_buffer = original_scratch_buffer
            return

        buffer_length = len(self.client.buffer) if self.client.buffer else 0
        if not buffer_length:
            logger.info("Buffer length is 0, skipping silence check")
            return
        logger.info(f"Buffer length: {buffer_length}")

        # Use default values if sampling_rate or samples_width are None
        sampling_rate = self.client.sampling_rate if getattr(self.client, 'sampling_rate', None) is not None else 1
        samples_width = self.client.samples_width if getattr(self.client, 'samples_width', None) is not None else 1
        bytes_per_second = sampling_rate * samples_width if sampling_rate and samples_width else 1
        logger.info(f"Bytes per second: {bytes_per_second}")
        buffer_duration = buffer_length / bytes_per_second if bytes_per_second else 0
        logger.info(f"Buffer duration: {buffer_duration}")

        # Check if vad_results has the expected structure
        if not isinstance(vad_results, list) or 'end' not in vad_results[-1]:
            logger.error("Invalid VAD results structure")
            self.client.scratch_buffer = original_scratch_buffer
            return

        if vad_results[-1]['end'] < (buffer_duration - self.silence_threshold_seconds):
            total_speech_duration = sum(segment.get('end', 0) - segment.get('start', 0) for segment in vad_results)
            if total_speech_duration >= self.min_speech_seconds:
                logger.info(f"Silence detected, processing {buffer_duration:.2f}s of audio")
                self.processing_flag = True
                self.client.buffer.clear()
                await self.process_audio_async(websocket, send_handle, asr_handle, debug_output)
            else:
                logger.info("Not enough speech detected, restoring buffer")
                self.client.scratch_buffer = original_scratch_buffer
        else:
            logger.info("No silence detected at buffer end, restoring buffer")
            self.client.scratch_buffer = original_scratch_buffer

    async def process_audio_async(self, websocket: WebSocket, vad_handle, asr_handle: DeploymentHandle, debug_output):
        try:
            start = time.time()
            logger.info("Starting asynchronous audio processing")
            transcription = await asr_handle.transcribe_raw.remote(client=self.client, debug_output=debug_output)
            
            if transcription is None:
                logger.error("Transcription returned None")
                return

            if hasattr(self.client, "increment_file_counter") and callable(self.client.increment_file_counter):
                self.client.increment_file_counter()
            else:
                logger.error("Client does not have a callable increment_file_counter method")
            
            if transcription.get('error'):
                logger.error(f"Transcription failed: {transcription['error']}")
                return

            # Use get to safely access 'text'
            if transcription.get('text', '').strip():
                end = time.time()
                logger.info(f"Transcription Processing time :{start-end}")
                transcription['processing_time'] = end - start
                json_transcription = json.dumps(transcription)
                await websocket.send_text(json_transcription)
                logger.info(f"Sent transcription: {json_transcription}")
            
            # Ensure scratch_buffer and audio properties are valid
            if hasattr(self.client, 'scratch_buffer') and self.client.scratch_buffer is not None:
                sampling_rate = self.client.sampling_rate if getattr(self.client, 'sampling_rate', None) is not None else 1
                samples_width = self.client.samples_width if getattr(self.client, 'samples_width', None) is not None else 1
                total_seconds = len(self.client.scratch_buffer) / (sampling_rate * samples_width)
                logger.debug(f"Processed {total_seconds:.2f}s of audio")
                self.client.scratch_buffer.clear()
            else:
                logger.error("Client scratch_buffer is None")
        except Exception as e:
            logger.error(f"Error during async processing: {e}")
        finally:
            logger.info("Finished async processing, resetting flag")
            self.processing_flag = False