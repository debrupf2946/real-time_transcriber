from .buffering_strategies import SilenceAtEndOfChunk

class BufferingStrategyFactory:
    """A factory class for creating instances of different buffering strategies."""

    @staticmethod
    def create_buffering_strategy(type, client, **kwargs):
        """create an instance of a buffering strategy based on the specified type."""

        if type == "silence_at_end_of_chunk":
            return SilenceAtEndOfChunk(client, **kwargs)
        else:
            raise ValueError(f"Unknown buffering strategy type: {type}")
