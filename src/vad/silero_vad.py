import io
from multiprocessing.reduction import send_handle
from datetime_utils import get_current_time_string_with_milliseconds
import torch
import torchaudio
import logging
import time

from .vad_interface import VADInterface
from audio_utils import convert_audio

logger = logging.getLogger("ray.serve")
logger.setLevel(logging.INFO)

from ray import serve


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
        #logger.info("Initializing SileroVAD")
        
        #logger.info("Loading Silero model from torch hub...")
        try:
            self.model, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
            #logger.info("Silero model loaded successfully")
            
            #logger.debug("Unpacking Silero utilities...")
            (self.get_speech_timestamps, _, self.read_audio, _, _) = self.utils
            #logger.debug("Utilities unpacked successfully")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(device)
            #logger.info(f"Model moved to {device}")
            
        except Exception as e:
            #logger.error(f"Error initializing Silero VAD: {str(e)}")
            raise

    async def detect_activity(self, client, debug_output):
        #logger.info("Starting voice activity detection with Silero")
        start_time = time.time()

        
        current_index = len(debug_output["silence_detection_timestamp"])

        debug_output["silence_detection_timestamp"].append({"silence_detection_index": current_index, "start_time": get_current_time_string_with_milliseconds(), "end_time": None, "vad_results": None})

        try:
            #logger.debug("Checking if audio buffer is empty...")
            if not client.scratch_buffer:
                #logger.warning("Received empty audio buffer from client.")
                return []

            #logger.debug("Converting audio buffer...")
            audio_data = await convert_audio(client.scratch_buffer)

            if not audio_data:
                #logger.warning("Audio conversion resulted in empty data.")
                return []

            #logger.debug("Loading audio data into tensor...")
            wav, sr = torchaudio.load(io.BytesIO(audio_data))

            if wav.size(1) == 0:
                #logger.warning("Loaded audio data is empty after torchaudio.load.")
                return []

            #logger.debug(f"Original audio shape: {wav.shape}, Sample rate: {sr}")

            # Convert multi-channel audio to mono
            if wav.dim() > 1 and wav.shape[0] > 1:
                #logger.debug("Converting multi-channel audio to mono...")
                wav = wav.mean(dim=0, keepdim=True)

            # Resample audio if necessary
            if sr != 16000:
                #logger.debug(f"Resampling audio from {sr}Hz to 16000Hz...")
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                wav = resampler(wav)

            if wav.size(1) == 0:
                #logger.warning("Audio data is empty after resampling.")
                return []

            #logger.info("Detecting speech timestamps...")
            speech_timestamps = self.get_speech_timestamps(wav, self.model, return_seconds=True)
            debug_output["silence_detection_timestamp"][current_index]["end_time"] = get_current_time_string_with_milliseconds()

            vad_segments = []
            if speech_timestamps:
                # Validate speech timestamp structure
                for segment in speech_timestamps:
                    if 'start' in segment and 'end' in segment:
                        vad_segments.append({
                            "start": float(segment['start']),
                            "end": float(segment['end']),
                            "confidence": 1.0
                        })
                    else:
                        #logger.warning(f"Invalid speech timestamp segment detected: {segment}")
                        pass

                #logger.info(f"Found {len(vad_segments)} voice segments")
                #logger.debug(f"Voice segments: {vad_segments}")

            #logger.debug(f"Speech detection completed in {time.time() - start_time:.2f} seconds")
            return vad_segments

        except Exception as e:
            #logger.error(f"Error in Silero VAD detection: {str(e)}")
            raise

