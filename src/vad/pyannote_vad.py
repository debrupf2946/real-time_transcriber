from os import remove
import os
import io
import torch
import torchaudio
import numpy as np


from pyannote.core import Segment
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection

from .vad_interface import VADInterface
from audio_utils import save_audio_to_file

from ray import serve
from ray.serve.handle import DeploymentHandle

@serve.deployment(
    ray_actor_options={"num_cpus": 1},
    autoscaling_config={"min_replicas": 1, "max_replicas":2},
)

# @serve.deployment(
#     ray_actor_options={"num_gpus": 1},
#     autoscaling_config={"min_replicas": 1, "max_replicas": 2},
# )
class PyannoteVAD(VADInterface):
    """
    Pyannote-based implementation of the VADInterface.
    """

    def __init__(self, **kwargs):
        """
        Initializes Pyannote's VAD pipeline.

        Args:
            model_name (str): The model name for Pyannote.
            auth_token (str, optional): Authentication token for Hugging Face.
        """
        
        model_name = kwargs.get('model_name', "pyannote/segmentation")

        # auth_token = os.environ.get('PYANNOTE_AUTH_TOKEN')
        auth_token = "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"
        if not auth_token:
            auth_token = kwargs.get('auth_token')
        
        if auth_token is None:
            raise ValueError("Missing required env var in PYANNOTE_AUTH_TOKEN or argument in --vad-args: 'auth_token'")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pyannote_args = kwargs.get('pyannote_args', {"onset": 0.3, "offset": 0.3, "min_duration_on": 0.3, "min_duration_off": 0.3})
        
        self.model = Model.from_pretrained(model_name, use_auth_token=auth_token)
        
        self.model.to(device)

        self.vad_pipeline = VoiceActivityDetection(segmentation=self.model)
        self.vad_pipeline.instantiate(pyannote_args)

    async def detect_activity(self, client):
        audio_file_path = await save_audio_to_file(client.scratch_buffer, client.get_file_name())
        print("audio file saved")
        vad_results = self.vad_pipeline(audio_file_path)
        
        # Check if file exists before deleting
        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)
        
        vad_segments = []
        if len(vad_results) > 0:
            vad_segments = [
                {"start": segment.start, "end": segment.end, "confidence": 1.0}
                for segment in vad_results.itersegments()
            ]
        return vad_segments
    
    async def detect_activity_byte_stream(self, client):
        audio_stream = io.BytesIO(client.scratch_buffer)
        print("audio file saved")
        vad_results = self.vad_pipeline(audio_file_path)
        
        # Check if file exists before deleting
        # if os.path.exists(audio_file_path):
        #     os.remove(audio_file_path)
        
        vad_segments = []
        if len(vad_results) > 0:
            vad_segments = [
                {"start": segment.start, "end": segment.end, "confidence": 1.0}
                for segment in vad_results.itersegments()
            ]
        return vad_segments
    
    
    
class Silero_VAD:
    def __init__(self, sampling_rate=16000, **kwargs):
        """
        Initialize Silero VAD model with default sampling rate of 16kHz
        
        Args:
            sampling_rate (int): Audio sampling rate. Silero VAD typically uses 16kHz
        """
        self.sampling_rate = sampling_rate
        self.model = self.load_silero_vad()
    
    def load_silero_vad(self):
        """
        Load the Silero VAD model
        
        Returns:
            torch.jit.ScriptModule: Loaded Silero VAD model
        """
        model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        return model
    
    def preprocess_audio(self, scratch_buffer):
        """
        Preprocess audio from scratch buffer for Silero VAD
        
        Args:
            scratch_buffer (bytes): Raw audio bytes
        
        Returns:
            torch.Tensor: Processed audio tensor
        """
        # Convert bytes to numpy array or torch tensor
        audio_array = np.frombuffer(scratch_buffer, dtype=np.float32)
        wav = torch.from_numpy(audio_array).float()
        
        # Reshape to match Silero model input (1, samples)
        wav = wav.unsqueeze(0)
        
        # Resample if needed
        if wav.size(1) > 0 and self.sampling_rate != 16000:
            transform = torchaudio.transforms.Resample(
                orig_freq=self.sampling_rate, 
                new_freq=16000
            )
            wav = transform(wav)
        
        return wav
    
    def detect_speech_timestamps(self, wav):
        """
        Detect speech timestamps using Silero VAD
        
        Args:
            wav (torch.Tensor): Processed audio tensor
        
        Returns:
            list: Speech timestamps in seconds
        """
        speech_timestamps = self.model(
            wav,
            sampling_rate=16000,  # Silero requires 16kHz
            return_seconds=True
        )
        return speech_timestamps
    
    async def detect_activity(self, client):
        """
        Detect speech activity in the client's audio stream
        
        Args:
            client: Client object with scratch_buffer attribute
        
        Returns:
            list: Speech timestamps
        """
        # Ensure the scratch buffer is not empty
        if not client.scratch_buffer:
            return []
        
        try:
            # Preprocess audio from scratch buffer
            wav = self.preprocess_audio(client.scratch_buffer)
            
            # Detect speech timestamps
            speech_timestamps = self.detect_speech_timestamps(wav)
            
            return speech_timestamps
        
        except Exception as e:
            print(f"Error in speech activity detection: {e}")
            return []

