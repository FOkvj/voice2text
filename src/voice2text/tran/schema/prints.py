from pydantic import BaseModel


class SampleInfo(BaseModel):
    """
    Represents information about a sample.
    """
    audio_duration: float
    audio_file_id: str
    created_at: str
    filename: str
    named: bool
    original_speaker: str
    sample_number: int
    speaker_id: str

