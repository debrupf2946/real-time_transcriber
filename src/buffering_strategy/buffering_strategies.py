# import os
# import asyncio
# import json
# import time
# from fastapi import WebSocket
# from audio_utils import save_audio_to_file

# from .buffering_strategy_interface import BufferingStrategyInterface
# from ray.serve.handle import DeploymentHandle

# import logging
# logger = logging.getLogger("ray.serve")
# logger.setLevel(logging.DEBUG)


# class SilenceAtEndOfChunk(BufferingStrategyInterface):
#     """
#     A buffering strategy that processes audio at the end of each chunk with silence detection.

#     This class is responsible for handling audio chunks, detecting silence at the end of each chunk,
#     and initiating the transcription process for the chunk.

#     Attributes:
#         client (Client): The client instance associated with this buffering strategy.
#         chunk_length_seconds (float): Length of each audio chunk in seconds.
#         chunk_offset_seconds (float): Offset time in seconds to be considered for processing audio chunks.
#     """

#     def __init__(self, client, **kwargs):
#         """
#         Initialize the SilenceAtEndOfChunk buffering strategy.

#         Args:
#             client (Client): The client instance associated with this buffering strategy.
#             **kwargs: Additional keyword arguments, including 'chunk_length_seconds' and 'chunk_offset_seconds'.
#         """
#         self.client = client

#         self.chunk_length_seconds = os.environ.get('BUFFERING_CHUNK_LENGTH_SECONDS')
#         if not self.chunk_length_seconds:
#             self.chunk_length_seconds = kwargs.get('chunk_length_seconds')
#         self.chunk_length_seconds = float(self.chunk_length_seconds)

#         self.chunk_offset_seconds = os.environ.get('BUFFERING_CHUNK_OFFSET_SECONDS')
#         if not self.chunk_offset_seconds:
#             self.chunk_offset_seconds = kwargs.get('chunk_offset_seconds')
#         self.chunk_offset_seconds = float(self.chunk_offset_seconds)

#         self.error_if_not_realtime = os.environ.get('ERROR_IF_NOT_REALTIME')
#         if not self.error_if_not_realtime:
#             self.error_if_not_realtime = kwargs.get('error_if_not_realtime', False)
        
#         self.processing_flag = False

#     def process_audio(self, websocket : WebSocket, vad_handle, asr_handle):
#         """
#         Process audio chunks by checking their length and scheduling asynchronous processing.

#         This method checks if the length of the audio buffer exceeds the chunk length and, if so,
#         it schedules asynchronous processing of the audio.

#         Args:
#             websocket (Websocket): The WebSocket connection for sending transcriptions.
#             vad_pipeline: The voice activity detection pipeline.
#             asr_pipeline: The automatic speech recognition pipeline.
#         """
#         chunk_length_in_bytes = self.chunk_length_seconds * self.client.sampling_rate * self.client.samples_width
#         if len(self.client.buffer) > chunk_length_in_bytes:
#             if self.processing_flag:
#                  logger.warning("Tried processing a new chunk while the previous one was still being processed")
#                 #  raise RuntimeError("Error in realtime processing: tried processing a new chunk while the previous one was still being processed")
#             else:
#                 self.client.scratch_buffer += self.client.buffer
#                 self.client.buffer.clear()
#                 self.processing_flag = True
#                 # Schedule the processing in a separate task
#                 asyncio.create_task(self.process_audio_async(websocket, vad_handle, asr_handle))
    
#     async def process_audio_async(self, websocket : WebSocket, vad_handle, asr_handle : DeploymentHandle):
#         """
#         Asynchronously process audio for activity detection and transcription.

#         This method performs heavy processing, including voice activity detection and transcription of
#         the audio data. It sends the transcription results through the WebSocket connection.

#         Args:
#             websocket (Websocket): The WebSocket connection for sending transcriptions.
#             vad_pipeline: The voice activity detection pipeline.
#             asr_pipeline: The automatic speech recognition pipeline.
#         """   
#         start = time.time()
#         vad_results = await vad_handle.detect_activity.remote(client = self.client)

#         if len(vad_results) == 0:
#             self.client.scratch_buffer.clear()
#             self.client.buffer.clear()
#             self.processing_flag = False
#             return

#         last_segment_should_end_before = ((len(self.client.scratch_buffer) / (self.client.sampling_rate * self.client.samples_width)) - self.chunk_offset_seconds)
#         if vad_results[-1]['end'] < last_segment_should_end_before:

#             transcription = await asr_handle.transcribe.remote(client = self.client)
#             self.client.increment_file_counter()
            
#             if transcription['text'] != '':
#                 end = time.time()
#                 transcription['processing_time'] = end - start
#                 json_transcription = json.dumps(transcription) 
#                 await websocket.send_text(json_transcription)
#             self.client.scratch_buffer.clear()
        
#         self.processing_flag = False
# #=================================================================================================================================
# import os
# import asyncio
# import json
# import time
# from fastapi import WebSocket
# from audio_utils import save_audio_to_file

# from .buffering_strategy_interface import BufferingStrategyInterface
# from ray.serve.handle import DeploymentHandle

# import logging
# logger = logging.getLogger("ray.serve")
# logger.setLevel(logging.DEBUG)


# class SilenceAtEndOfChunk(BufferingStrategyInterface):
#     """
#     A buffering strategy that processes audio only when silence is detected at the end of the buffer.

#     This class is responsible for handling audio chunks, appending them to a buffer,
#     detecting silence at the end of the buffer, and initiating the transcription 
#     process when silence is detected.

#     Attributes:
#         client (Client): The client instance associated with this buffering strategy.
#         silence_threshold_seconds (float): Minimum duration of silence (in seconds) to trigger processing.
#         min_speech_seconds (float): Minimum duration of speech required before processing.
#     """

#     def __init__(self, client, **kwargs):
#         """
#         Initialize the SilenceAtEndOfChunk buffering strategy.

#         Args:
#             client (Client): The client instance associated with this buffering strategy.
#             **kwargs: Additional keyword arguments including 'silence_threshold_seconds' and 'min_speech_seconds'.
#         """
#         logger.info(f"initialized Silence end of chunk")
#         self.client = client

#         self.silence_threshold_seconds = os.environ.get('BUFFERING_SILENCE_THRESHOLD_SECONDS')
#         if not self.silence_threshold_seconds:
#             self.silence_threshold_seconds = kwargs.get('silence_threshold_seconds', 0.5)
#         self.silence_threshold_seconds = float(self.silence_threshold_seconds)

#         self.min_speech_seconds = os.environ.get('BUFFERING_MIN_SPEECH_SECONDS')
#         if not self.min_speech_seconds:
#             self.min_speech_seconds = kwargs.get('min_speech_seconds', 0.5)
#         self.min_speech_seconds = float(self.min_speech_seconds)

#         self.max_buffer_seconds = os.environ.get('BUFFERING_MAX_BUFFER_SECONDS')
#         if not self.max_buffer_seconds:
#             self.max_buffer_seconds = kwargs.get('max_buffer_seconds', 30.0)
#         self.max_buffer_seconds = float(self.max_buffer_seconds)

#         self.error_if_not_realtime = os.environ.get('ERROR_IF_NOT_REALTIME')
#         if not self.error_if_not_realtime:
#             self.error_if_not_realtime = kwargs.get('error_if_not_realtime', False)
        
#         self.processing_flag = False
#         self.last_chunk_time = time.time()

#     def process_audio(self, websocket: WebSocket, vad_handle, asr_handle):
#         """
#         Process audio chunks by checking for silence at the end of the buffer.

#         This method is called when a new audio chunk is received. It checks if there is
#         silence at the end of the buffer, and if so, schedules asynchronous processing.

#         Args:
#             websocket (Websocket): The WebSocket connection for sending transcriptions.
#             vad_handle: The voice activity detection pipeline handle.
#             asr_handle: The automatic speech recognition pipeline handle.
#         """
#         logger.info(f"started processing audio")
#         self.last_chunk_time = time.time()
        
#         # Check if we're already processing
#         if self.processing_flag:
#             logger.debug("Already processing a chunk, skipping check")
#             return
            
#         # Check if buffer is too large (safety mechanism)
#         max_buffer_size_bytes = self.max_buffer_seconds * self.client.sampling_rate * self.client.samples_width
#         if len(self.client.buffer) > max_buffer_size_bytes:
#             logger.warning(f"Buffer exceeded maximum size of {self.max_buffer_seconds} seconds, forcing processing")
#             self.processing_flag = True
#             asyncio.create_task(self.process_audio_async(websocket, vad_handle, asr_handle))
#             return
            
#         # Only check for silence if we have enough data
#         min_buffer_size = self.silence_threshold_seconds * self.client.sampling_rate * self.client.samples_width
#         if len(self.client.buffer) < min_buffer_size:
#             return
            
#         # Schedule task to check for silence detection
#         logger.info(f"checking for silence")
#         asyncio.create_task(self.check_silence_and_process(websocket, vad_handle, asr_handle))
            
#     async def check_silence_and_process(self, websocket: WebSocket, vad_handle, asr_handle):
#         """
#         Check for silence at the end of the buffer and process if detected.
        
#         Args:
#             websocket (Websocket): The WebSocket connection for sending transcriptions.
#             vad_handle: The voice activity detection pipeline handle.
#             asr_handle: The automatic speech recognition pipeline handle.
#         """
#         logger.info(f"entered check_silence_and_process function")
#         if self.processing_flag:
#             return
            
#         # Use the current buffer directly for VAD
#         # We can save the original buffer content temporarily
#         original_scratch_buffer = self.client.scratch_buffer
#         self.client.scratch_buffer = bytearray(self.client.buffer)
        
#         # Detect silence using VAD
#         vad_results = await vad_handle.detect_activity.remote(client=self.client)
        
#         # If no speech detected at all, restore and return
#         if len(vad_results) == 0:
#             self.client.scratch_buffer = original_scratch_buffer
#             return
            
#         # Get buffer duration in seconds
#         buffer_duration = len(self.client.buffer) / (self.client.sampling_rate * self.client.samples_width)
        
#         # Check if the last speech segment ends before the silence threshold
#         if vad_results[-1]['end'] < (buffer_duration - self.silence_threshold_seconds):
#             # Check if we have enough speech
#             total_speech_duration = sum(segment['end'] - segment['start'] for segment in vad_results)
            
#             if total_speech_duration >= self.min_speech_seconds:
#                 logger.info(f"Silence detected at end of buffer, processing {buffer_duration:.2f}s of audio")
#                 self.processing_flag = True
#                 # We already have the buffer in scratch_buffer, so clear the buffer
#                 self.client.buffer.clear()
#                 # Process the audio
#                 await self.process_audio_async(websocket, vad_handle, asr_handle)
#             else:
#                 # Not enough speech, restore original scratch buffer
#                 self.client.scratch_buffer = original_scratch_buffer
#         else:
#             # No silence at the end, restore original scratch buffer
#             self.client.scratch_buffer = original_scratch_buffer
    
#     async def process_audio_async(self, websocket: WebSocket, vad_handle, asr_handle: DeploymentHandle):
#         """
#         Asynchronously process audio for transcription.

#         This method performs the actual transcription of audio data and
#         sends the transcription results through the WebSocket connection.
#         It assumes that voice activity detection has already been performed.

#         Args:
#             websocket (Websocket): The WebSocket connection for sending transcriptions.
#             vad_handle: The voice activity detection pipeline handle (not used here).
#             asr_handle: The automatic speech recognition pipeline handle.
#         """   
#         try:
#             start = time.time()
            
#             # Directly transcribe the audio since we've already confirmed speech activity
#             transcription = await asr_handle.batch_transcribe.remote(client=self.client)
#             self.client.increment_file_counter()
            
#             if transcription['text'].strip() != '':
#                 end = time.time()
#                 transcription['processing_time'] = end - start
#                 json_transcription = json.dumps(transcription) 
#                 await websocket.send_text(json_transcription)
                
#             logger.debug(f"Processed {len(self.client.scratch_buffer) / (self.client.sampling_rate * self.client.samples_width):.2f}s of audio")
#             self.client.scratch_buffer.clear()
#         except Exception as e:
#             logger.error(f"Error processing audio: {str(e)}")
#         finally:
#             self.processing_flag = False
# =================================================------------------------========================================================
import os
import asyncio
import json
import time
from fastapi import WebSocket
from audio_utils import save_audio_to_file

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
        buffer_start_time=time.time()
        vad_results = await vad_handle.detect_activity.remote(client=self.client)
        buffer_end_time=time.time()
        logger.debug(f"VAD results: {vad_results}")
        logger.info(f"VAD processing time{buffer_start_time-buffer_end_time}")

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
                logger.info("starting process audio")
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
            transcription = await asr_handle.transcribe_byte_stream.remote(client=self.client)
            self.client.increment_file_counter()
            
            
            if transcription.get('error'):
                logger.error(f"Transcription failed: {transcription['error']}")
                return

            if transcription['text'].strip():
                end = time.time()
                logger.info(f"Transcription Processing time :{start-end}")
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

