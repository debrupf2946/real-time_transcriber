from .faster_whisper_asr import FasterWhisperASR

class ASRFactory:
    @staticmethod
    def create_asr_pipeline(type, **kwargs):
        if type == "faster_whisper":
            return FasterWhisperASR.bind(**kwargs)
        else:
            raise ValueError(f"Unknown ASR pipeline type: {type}")
