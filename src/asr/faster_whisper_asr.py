import os
from faster_whisper import WhisperModel, BatchedInferencePipeline

from .asr_interface import ASRInterface
from ..audio_utils import save_audio_to_file

import uuid
from ray import serve

import logging
logger = logging.getLogger("ray.serve")
logger.setLevel(logging.DEBUG)

@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={"min_replicas": 1, "max_replicas": 2},
)

# "deepdml/faster-whisper-large-v3-turbo-ct2"
class FasterWhisperASR(ASRInterface):
    def __init__(self, **kwargs):
        model_size = kwargs.get('model_size', "deepdml/faster-whisper-large-v3-turbo-ct2")
        # Run on GPU with FP16
        self.asr_pipeline = WhisperModel(
            model_size, device="cuda", compute_type="float16")
        
    async def batch_transcribe(self,client):
        file_path = await save_audio_to_file(client.scratch_buffer, client.get_file_name())

        # language = None if client.config['language'] is None else language_codes.get(
        #     client.config['language'].lower())
        language = "en"
        
        self.batched_model = BatchedInferencePipeline(model=self.asr_pipeline)

        segments, info = self.batched_model.transcribe(
            file_path, word_timestamps=True, language="en",batch_size=16,beam_size=2)

        segments = list(segments)  # The transcription will actually run here.
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
            ]
        }
        return to_return

        

    async def transcribe(self, language, scratch_buffer):

        logger.info(f"Transcribing audio file: {language}")
        file_name = str(uuid.uuid4()) + ".wav"
        logger.info(f"Transcribing audio file: {file_name}")
        file_path = await save_audio_to_file(scratch_buffer, file_name) 
        logger.info(f"File path: {file_path}")
        try:
            segments, info = self.asr_pipeline.transcribe(
                file_path, word_timestamps=True, language=language, beam_size=2)
            logger.info(f"Segments: {segments}")
            segments = list(segments)  # The transcription will actually run here.
            logger.info(f"Segments after list: {segments}")
        except Exception as e:
            print(f"Error transcribing audio: {e}")
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
        flattened_words = [
            word for segment in segments for word in segment.words]
        return {
            "language": info.language,
            "language_probability": info.language_probability,
            "text": ' '.join([s.text.strip() for s in segments]),
            "words":
            [
                {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability} for w in flattened_words
            ]
        }
