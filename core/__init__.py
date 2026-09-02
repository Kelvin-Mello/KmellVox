"""KmellVox Core Module - Dublagem, Tradução, Narração e Lip Sync com IA."""

from .hardware import (
    detect_gpu_profile,
    detect_hardware,
    HardwareProfile,
    HardwareInfo,
    ModelProfile,
)
from .audio_extract import extract_audio, get_audio_info
from .transcribe import (
    Transcriber,
    WhisperTranscriber,
    TranscriptionResult,
    TranscriptionSegment,
    format_segments_to_srt,
)
from .translate import (
    Translator,
    LLMTranslator,
    TranslatedSegment,
    TranslationResult,
)
from .voice_clone import (
    F5TTSEngine,
    IndexTTS2Engine,
    get_tts_engine,
    BaseTTSEngine,
    RhythmControlConfig,
    ClonedAudioSegment,
    VoiceCloner,
    VoiceCloneConfig,
)
from .lipsync import (
    LipSyncEngine,
    LipSyncer,
    LipSyncConfig,
    LipSyncResult,
)
from .assemble import (
    burn_subtitles,
    mux_audio_video,
    export_raw_package,
    assemble_final_video,
    AssemblyConfig,
)
from .pipeline import (
    DubPipeline,
    DubbingPipeline,
    PipelineConfig,
    PipelineProgress,
)
from .narration import (
    detect_text_format,
    parse_srt,
    list_preset_voices,
    NarrationJob,
    NarrationEngine,
)

__all__ = [
    "detect_gpu_profile",
    "detect_hardware",
    "HardwareProfile",
    "HardwareInfo",
    "ModelProfile",
    "extract_audio",
    "get_audio_info",
    "Transcriber",
    "WhisperTranscriber",
    "TranscriptionResult",
    "TranscriptionSegment",
    "format_segments_to_srt",
    "Translator",
    "LLMTranslator",
    "TranslatedSegment",
    "TranslationResult",
    "F5TTSEngine",
    "IndexTTS2Engine",
    "get_tts_engine",
    "BaseTTSEngine",
    "RhythmControlConfig",
    "ClonedAudioSegment",
    "VoiceCloner",
    "VoiceCloneConfig",
    "LipSyncEngine",
    "LipSyncer",
    "LipSyncConfig",
    "LipSyncResult",
    "burn_subtitles",
    "mux_audio_video",
    "export_raw_package",
    "assemble_final_video",
    "AssemblyConfig",
    "DubPipeline",
    "DubbingPipeline",
    "PipelineConfig",
    "PipelineProgress",
    "detect_text_format",
    "parse_srt",
    "list_preset_voices",
    "NarrationJob",
    "NarrationEngine",
]
