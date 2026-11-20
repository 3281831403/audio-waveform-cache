"""Waveform peak-cache public API."""

from .peaks import (
    SAMPLE_RATE,
    TOLERANCE,
    WaveformError,
    bucket_peaks,
    bucket_rms,
    incremental_update,
    merge_levels,
    multi_resolution,
    normalize_peaks,
    peaks_batch,
    query_range,
)

__all__ = [
    "SAMPLE_RATE",
    "TOLERANCE",
    "WaveformError",
    "bucket_peaks",
    "bucket_rms",
    "incremental_update",
    "merge_levels",
    "multi_resolution",
    "normalize_peaks",
    "peaks_batch",
    "query_range",
]
