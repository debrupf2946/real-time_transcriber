class BufferingStrategyInterface:
    """An interface class for buffering strategies in audio processing systems."""

    def process_audio(self, websocket, vad_pipeline, asr_pipeline):
        """Process audio data using the given WebSocket connection, VAD pipeline, and ASR pipeline."""
        
        raise NotImplementedError("This method should be implemented by subclasses.")
