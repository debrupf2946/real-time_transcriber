from os import remove
import os
import io
import torch
import torchaudio
import numpy as np
import logging
import time

from pyannote.core import Segment
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection

from .vad_interface import VADInterface
from audio_utils import save_audio_to_file,convert_audio

logger = logging.getLogger("ray.serve")
logger.setLevel(logging.DEBUG)

from ray import serve
from ray.serve.handle import DeploymentHandle

# @serve.deployment(
#     ray_actor_options={"num_cpus": 1},
#     autoscaling_config={"min_replicas": 1, "max_replicas":2},
# )
# class PyannoteVAD(VADInterface):
#     """
#     Pyannote-based implementation of the VADInterface.
#     """

#     def __init__(self, **kwargs):
#         """
#         Initializes Pyannote's VAD pipeline.

#         Args:
#             model_name (str): The model name for Pyannote.
#             auth_token (str, optional): Authentication token for Hugging Face.
#         """
#         logger.info("Initializing PyannoteVAD")
        
#         model_name = kwargs.get('model_name', "pyannote/segmentation")
#         logger.debug(f"Using model: {model_name}")

#         auth_token = "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"
#         if not auth_token:
#             auth_token = kwargs.get('auth_token')
#             logger.debug("Using auth token from kwargs")
        
#         if auth_token is None:
#             logger.error("Missing authentication token")
#             raise ValueError("Missing required env var in PYANNOTE_AUTH_TOKEN or argument in --vad-args: 'auth_token'")
        
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         logger.info(f"Using device: {device}")
        
#         pyannote_args = kwargs.get('pyannote_args', {"onset": 0.3, "offset": 0.3, "min_duration_on": 0.3, "min_duration_off": 0.3})
#         logger.debug(f"Pyannote arguments: {pyannote_args}")
        
#         logger.info("Loading Pyannote model...")
#         self.model = Model.from_pretrained(model_name, use_auth_token=auth_token)
#         logger.info("Model loaded successfully")
        
#         self.model.to(device)
#         logger.debug(f"Model moved to {device}")

#         logger.info("Instantiating VAD pipeline...")
#         self.vad_pipeline = VoiceActivityDetection(segmentation=self.model)
#         self.vad_pipeline.instantiate(pyannote_args)
#         logger.info("VAD pipeline ready")

#     async def detect_activity(self, client):
#         logger.info("Starting voice activity detection")
#         start_time = time.time()
        
#         logger.debug("Saving audio to file...")
#         audio_file_path = await save_audio_to_file(client.scratch_buffer, client.get_file_name())
#         logger.debug(f"Audio saved to: {audio_file_path}")
        
#         logger.info("Running VAD pipeline...")
#         vad_results = self.vad_pipeline(audio_file_path)
#         logger.debug(f"VAD pipeline completed in {time.time() - start_time:.2f} seconds")
        
#         if os.path.exists(audio_file_path):
#             logger.debug(f"Cleaning up temporary file: {audio_file_path}")
#             os.remove(audio_file_path)
        
#         vad_segments = []
#         if len(vad_results) > 0:
#             vad_segments = [
#                 {"start": segment.start, "end": segment.end, "confidence": 1.0}
#                 for segment in vad_results.itersegments()
#             ]
#             logger.info(f"Found {len(vad_segments)} voice segments")
#             logger.debug(f"Voice segments: {vad_segments}")
#         else:
#             logger.info("No voice segments detected")
            
#         return vad_segments
    

@serve.deployment(
    ray_actor_options={"num_cpus": 1},
    autoscaling_config={"min_replicas": 1, "max_replicas":1},
)
class SileroVAD(VADInterface):
    """
    Silero-based implementation of the VADInterface.
    """

    def __init__(self, **kwargs):
        """
        Initializes Silero VAD model.

        Args:
            **kwargs: Additional arguments for the VAD pipeline creation.
        """
        logger.info("Initializing SileroVAD")
        
        logger.info("Loading Silero model from torch hub...")
        try:
            self.model, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
            logger.info("Silero model loaded successfully")
            
            logger.debug("Unpacking Silero utilities...")
            (self.get_speech_timestamps, _, self.read_audio, _, _) = self.utils
            logger.debug("Utilities unpacked successfully")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(device)
            logger.info(f"Model moved to {device}")
            
        except Exception as e:
            logger.error(f"Error initializing Silero VAD: {str(e)}")
            raise

    async def detect_activity(self, client):
        logger.info("Starting voice activity detection with Silero")
        start_time = time.time()
        
        try:
            logger.debug("Converting audio buffer to byte array...")
            # audio_data = np.frombuffer(client.scratch_buffer, dtype=np.float32)
            # audio_data=bytearray(client.scratch_buffer)
            # logger.debug(f"Audio data shape: {audio_data.shape}")
            
            logger.debug("Converting to torch tensor...")
            # audio_tensor = torch.from_numpy(io.BytesIO(audio_data)).unsqueeze(0)
            # wav, sr=torchaudio.load(io.BytesIO(audio_data))
            # logger.debug(f"Audio tensor shape: {audio_tensor.shape}")
            
            # if len(wav.shape) > 1 and wav.shape[0] > 1:  # Check if multi-channel
            #     wav = wav.mean(dim=0, keepdim=True)
                        
            # if wav.size(1) > 0 and sr != 16000:
            #     transform = torchaudio.transforms.Resample(
            #         orig_freq=sr,
            #         new_freq=16000
            #     )
            #     wav = transform(wav)
            
            audio_data=await convert_audio(client.scratch_buffer)
            wav, sr=torchaudio.load(io.BytesIO(audio_data))
            logger.info("Detecting speech timestamps...")
            speech_timestamps = self.get_speech_timestamps(wav, self.model, return_seconds=True)
            logger.debug(f"Speech detection completed in {time.time() - start_time:.2f} seconds")
            
            vad_segments = []
            if speech_timestamps:
                vad_segments = [
                    {"start": float(segment['start']), "end": float(segment['end']), "confidence": 1.0}
                    for segment in speech_timestamps
                ]
                logger.info(f"Found {len(vad_segments)} voice segments")
                logger.debug(f"Voice segments: {vad_segments}")
            else:
                logger.info("No voice segments detected")
                
            return vad_segments
            
        except Exception as e:
            logger.error(f"Error in Silero VAD detection: {str(e)}")
            raise
    

