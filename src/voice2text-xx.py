import os
import pickle
import numpy as np
import librosa
import whisper
from resemblyzer import VoiceEncoder, preprocess_wav
from  pyannote.audio.pipelines import SpeakerDiarization
# Default paths as constants
DEFAULT_MODEL_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "whisper_models")
DEFAULT_ALIGN_MODEL_DIR = "/Users/dongzhancai1/Desktop/voice2text/src/models"
DEFAULT_VOICE_PRINTS_PATH = os.path.join(os.path.expanduser("~"), ".cache", "voice_prints.pkl")


class AudioProcessor:
    """Handles audio file processing operations"""

    @staticmethod
    def process_mp3(mp3_path):
        """Process MP3 file and return audio data suitable for voice print extraction"""
        wav, sr = librosa.load(mp3_path, sr=16000, mono=True)
        return wav, sr


class ModelManager:
    """Manages loading and initialization of all required models"""

    def __init__(self, model_name="base", device="cpu",
                 model_cache_dir=DEFAULT_MODEL_CACHE_DIR,
                 align_model_dir=DEFAULT_ALIGN_MODEL_DIR,
                 align_model_name=None,
                 diarize_model_path=None):
        self.model_name = model_name
        self.device = device
        self.model_cache_dir = model_cache_dir
        self.align_model_dir = align_model_dir
        self.align_model_name = align_model_name or os.path.join(
            self.align_model_dir, "wav2vec2-large-xlsr-53-chinese-zh-cn")
        self.diarize_model_path = diarize_model_path or os.path.join(
            self.align_model_dir, "speaker-diarization-3.1")
        self.diarize_config_path = os.path.join(self.diarize_model_path, "config.yaml")

        self.whisper_model = None
        self.align_model = None
        self.align_metadata = None
        self.diarize_model = None
        self.voice_encoder = None

        # Ensure directories exist
        os.makedirs(self.model_cache_dir, exist_ok=True)
        os.makedirs(self.align_model_dir, exist_ok=True)

    def load_all_models(self):
        """Load all necessary models"""
        print(f"Loading Whisper model: {self.model_name} from cache if available...")
        self._load_whisper_model()
        self._load_align_model()
        self._load_diarize_model()
        self._load_voice_encoder()
        return (self.whisper_model, self.align_model, self.align_metadata,
                self.diarize_model, self.voice_encoder)

    def _load_whisper_model(self):
        """Load the Whisper speech recognition model"""
        self.whisper_model = whisper.load_model(
            self.model_name,
            download_root=self.model_cache_dir
        ).to(self.device)

    def _load_align_model(self):
        """Load the alignment model for improving timestamp accuracy"""
        print(f"Loading alignment model from: {self.align_model_name}")
        try:
            from whisperx import load_align_model
            self.align_model, self.align_metadata = load_align_model(
                "zh", device=self.device, model_name=self.align_model_name
            )
        except Exception as e:
            print(f"Error loading alignment model: {e}")
            self.align_model, self.align_metadata = None, None

    def _load_diarize_model(self):
        """Load the speaker diarization model"""
        print(f"Loading diarization model from: {self.diarize_model_path}")
        try:
            # First attempt: Use WhisperX diarization
            from whisperx.diarize import DiarizationPipeline
            self.diarize_model = DiarizationPipeline(model_name=self.diarize_config_path)

            # Test if model loaded correctly
            if self.diarize_model is None:
                raise Exception("DiarizationPipeline returned None")

        except Exception as e:
            print(f"Error loading DiarizationPipeline: {e}")
            print("Falling back to direct PyAnnote API...")

            # Second attempt: Use PyAnnote directly
            try:
                from pyannote.audio import Pipeline
                import torch

                self.diarize_model = Pipeline.from_pretrained(self.diarize_model_path)

                if self.diarize_model is None:
                    raise Exception("Pipeline.from_pretrained returned None")

                # Manually set device
                if self.device != "cpu":
                    self.diarize_model.to(torch.device(self.device))

            except Exception as e:
                print(f"Error in fallback approach: {e}")
                raise Exception(f"Failed to load diarization model: {e}")

    def _load_voice_encoder(self):
        """Load the voice encoder model for speaker verification"""
        print("Loading voice encoder model for speaker verification...")
        self.voice_encoder = VoiceEncoder(device=self.device)


class VoicePrintManager:
    """Manages voice print registration and identification"""

    def __init__(self, voice_encoder, voice_prints_path=DEFAULT_VOICE_PRINTS_PATH):
        self.voice_encoder = voice_encoder
        self.voice_prints_path = voice_prints_path
        self.voice_prints = self._load_voice_prints()

    def _load_voice_prints(self):
        """Load existing voice prints or create empty dict if none exists"""
        if os.path.exists(self.voice_prints_path):
            with open(self.voice_prints_path, 'rb') as f:
                return pickle.load(f)
        return {}

    def save_voice_prints(self):
        """Save current voice prints to disk"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.voice_prints_path), exist_ok=True)
        with open(self.voice_prints_path, 'wb') as f:
            pickle.dump(self.voice_prints, f)

    def register_voice(self, person_name, mp3_file_path):
        """Register a new voice print from an MP3 file"""
        print(f"Registering voice for: {person_name} from MP3 file")

        # Process MP3 file
        wav, _ = AudioProcessor.process_mp3(mp3_file_path)

        # Preprocess audio and extract voice embedding
        wav = preprocess_wav(wav)
        embedding = self.voice_encoder.embed_utterance(wav)

        # Add to voice prints
        self.voice_prints[person_name] = embedding
        self.save_voice_prints()

        print(f"Voice print for {person_name} registered successfully.")
        return self.voice_prints

    def register_voices_from_directory(self, directory_path):
        """Register voice prints from all audio files in a directory

        The filename (without extension) is used as the speaker name.
        Supports mp3, wav, m4a, and other common audio formats.
        """
        print(f"Registering voice prints from directory: {directory_path}")

        if not os.path.isdir(directory_path):
            print(f"Error: {directory_path} is not a valid directory")
            return self.voice_prints

        # List of supported audio extensions
        audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']
        registered_count = 0

        # Process each audio file in the directory
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)

            # Skip directories and non-audio files
            if os.path.isdir(file_path):
                continue

            _, ext = os.path.splitext(filename)
            if ext.lower() not in audio_extensions:
                continue

            # Use filename without extension as speaker name
            speaker_name = os.path.splitext(filename)[0]

            try:
                # Register the voice
                self.register_voice(speaker_name, file_path)
                registered_count += 1
            except Exception as e:
                print(f"Error registering {speaker_name}: {e}")

        print(f"Successfully registered {registered_count} voice prints from directory")
        return self.voice_prints

    def identify_speakers(self, diarize_segments, mp3_file_path, threshold=0.7):
        """Identify speakers based on registered voice prints"""
        print("Identifying speakers based on registered voice prints...")

        # If no voice prints exist, return unknown speakers
        if not self.voice_prints:
            print("No registered voice prints found.")
            unique_speakers = diarize_segments['speaker'].unique()
            return {speaker: f"Unknown_Speaker_{i}" for i, speaker in enumerate(unique_speakers)}

        # Process MP3 file
        wav, sr = AudioProcessor.process_mp3(mp3_file_path)

        # Dictionary to store speaker identities
        speaker_identities = {}

        # Process each unique speaker
        for speaker in diarize_segments['speaker'].unique():
            # Get all segments for this speaker
            speaker_segments = diarize_segments[diarize_segments['speaker'] == speaker]

            # Collect all audio segments for this speaker
            speaker_wavs = []
            print(speaker_segments)

            for _, segment in speaker_segments.iterrows():
                start = segment['start']
                end = segment['end']

                # Extract time segment
                start_sample = int(start * sr)
                end_sample = min(int(end * sr), len(wav))

                if end_sample > start_sample:
                    segment_audio = wav[start_sample:end_sample]
                    speaker_wavs.append(segment_audio)

            if speaker_wavs:
                # Combine all segments for better voice print
                combined_wav = np.concatenate(speaker_wavs)

                # Ensure enough audio data for reliable voice print
                min_samples = sr * 1.5  # at least 1.5 seconds
                if len(combined_wav) >= min_samples:
                    processed_wav = preprocess_wav(combined_wav)
                    embedding = self.voice_encoder.embed_utterance(processed_wav)

                    # Compare with registered voice prints
                    best_match = None
                    best_score = 0

                    for person_name, registered_embedding in self.voice_prints.items():
                        similarity = np.inner(embedding, registered_embedding)
                        print(f"Comparing {speaker} with {person_name}: similarity = {similarity:.2f}")
                        if similarity > best_score:
                            best_score = similarity
                            best_match = person_name

                    # Assign identity if similarity above threshold
                    if best_score >= threshold:
                        speaker_identities[speaker] = best_match
                    else:
                        speaker_identities[speaker] = f"Unknown:{speaker} (similarity: {best_score:.2f})"
                else:
                    speaker_identities[speaker] = f"TooShort_{speaker}"
            else:
                speaker_identities[speaker] = f"NoValidAudio_{speaker}"

        return speaker_identities

    def list_registered_voices(self):
        """List all registered voice prints"""
        if not self.voice_prints:
            print("No voice prints registered.")
            return

        print(f"Currently registered voice prints ({len(self.voice_prints)}):")
        for i, name in enumerate(self.voice_prints.keys(), 1):
            print(f"  {i}. {name}")


class Transcriber:
    """Main class for transcribing audio with speaker identification"""

    def __init__(self, whisper_model_name="base", device="cpu",
                 model_manager=None, voice_print_manager=None,
                 model_cache_dir=DEFAULT_MODEL_CACHE_DIR,
                 align_model_dir=DEFAULT_ALIGN_MODEL_DIR,
                 align_model_name=None,
                 diarize_model_path=None,
                 voice_prints_path=DEFAULT_VOICE_PRINTS_PATH):
        """Initialize with optional dependency injection for managers"""

        # Create model manager if not provided
        if model_manager is None:
            model_manager = ModelManager(
                model_name=whisper_model_name,
                device=device,
                model_cache_dir=model_cache_dir,
                align_model_dir=align_model_dir,
                align_model_name=align_model_name,
                diarize_model_path=diarize_model_path
            )
            # Load all models
            self.models = model_manager.load_all_models()
            self.whisper_model, self.align_model, self.align_metadata, self.diarize_model, self.voice_encoder = self.models
        else:
            # Use provided model manager
            self.model_manager = model_manager
            self.whisper_model = model_manager.whisper_model
            self.align_model = model_manager.align_model
            self.align_metadata = model_manager.align_metadata
            self.diarize_model = model_manager.diarize_model
            self.voice_encoder = model_manager.voice_encoder

        # Create voice print manager if not provided
        if voice_print_manager is None and self.voice_encoder is not None:
            self.voice_print_manager = VoicePrintManager(
                self.voice_encoder,
                voice_prints_path=voice_prints_path
            )
        else:
            self.voice_print_manager = voice_print_manager

    def transcribe_file(self, mp3_file_path):
        """Transcribe an MP3 file with speaker identification"""
        print(f"Transcribing MP3 file: {mp3_file_path}...")

        # Use Whisper for initial transcription
        result = self.whisper_model.transcribe(mp3_file_path, language="zh")
        segments = result["segments"]

        # Use WhisperX for timestamp alignment
        from whisperx import align
        aligned_segments = align(segments, self.align_model, self.align_metadata,
                                 mp3_file_path, device="cpu")

        # Perform speaker diarization
        diarize_segments = self.diarize_model(mp3_file_path)

        # Identify speakers
        speaker_identities = self.voice_print_manager.identify_speakers(
            diarize_segments, mp3_file_path
        )

        # Combine speaker info with transcription
        speaker_segments = []

        for segment in aligned_segments['segments']:
            # Find speaker for this segment
            segment_start = segment["start"]
            segment_end = segment["end"]
            speaker_id = self._find_speaker(segment_start, segment_end, diarize_segments)

            # Get speaker identity
            speaker_name = speaker_identities.get(speaker_id, f"Unknown_Speaker_{speaker_id}")

            # Format output
            start_time = self._format_time(segment["start"])
            end_time = self._format_time(segment["end"])
            text = segment["text"]
            speaker_segments.append(f"[{start_time}-{end_time}] [{speaker_name}] {text}")

        # Join segments into single string
        transcript = "\n".join(speaker_segments)

        # Save output to file
        output_file = mp3_file_path.rsplit(".", 1)[0] + "_transcript.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(transcript)
        print(f"Transcription saved to {output_file}")

        return transcript

    def register_voice(self, person_name, mp3_file_path):
        """Register a new voice print"""
        return self.voice_print_manager.register_voice(person_name, mp3_file_path)

    def register_voices_from_directory(self, directory_path):
        """Register voice prints from all audio files in a directory"""
        return self.voice_print_manager.register_voices_from_directory(directory_path)

    def list_registered_voices(self):
        """List all registered voice prints"""
        return self.voice_print_manager.list_registered_voices()

    def _find_speaker(self, start_time, end_time, diarize_segments):
        """Find the speaker with maximum overlap for a given time segment"""
        max_overlap = 0
        best_speaker = None

        # Check all diarization segments
        for _, segment in diarize_segments.iterrows():
            segment_start = segment['start']
            segment_end = segment['end']
            speaker = segment['speaker']

            # Calculate overlap
            overlap_start = max(start_time, segment_start)
            overlap_end = min(end_time, segment_end)
            overlap = max(0, overlap_end - overlap_start)

            if overlap >= max_overlap:
                max_overlap = overlap
                best_speaker = speaker

        return best_speaker if best_speaker else "Unknown"

    @staticmethod
    def _format_time(seconds):
        """Format seconds to HH:MM:SS format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"


def main():
    """Main function to demonstrate usage"""
    # Configuration
    wishper_model_name = "base"
    audio_file = "data/家有儿女吃饭.mp3"
    device = "cpu"  # Change to "cuda" if GPU available

    # Base model paths
    model_base_dir = "/Users/dongzhancai1/Desktop/voice2text/src/models"

    # Create transcriber with custom model paths
    transcriber = Transcriber(
        whisper_model_name=wishper_model_name,
        device=device,
        model_cache_dir=os.path.join(os.path.expanduser("~"), ".cache", "whisper_models"),
        align_model_dir=model_base_dir,
        align_model_name=os.path.join(model_base_dir, "wav2vec2-large-xlsr-53-chinese-zh-cn"),
        diarize_model_path=os.path.join(model_base_dir, "speaker-diarization-3.1"),
        voice_prints_path=os.path.join(os.path.expanduser("~"), ".cache", "voice_prints.pkl")
    )

    # List registered voices
    transcriber.list_registered_voices()

    # Register individual voices (uncomment as needed)
    # transcriber.register_voice("刘星", "data/sample/刘星.mp3")
    # transcriber.register_voice("小雪", "data/sample/小雪.mp3")

    # Or register all voices from a directory at once
    # transcriber.register_voices_from_directory("data/sample")

    # Transcribe audio file
    transcript = transcriber.transcribe_file(audio_file)

    # Print result
    print("Transcription with Timestamps and Speaker Identification:")
    print(transcript)


if __name__ == "__main__":
    main()