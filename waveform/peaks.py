"""Fixed-resolution waveform peak-cache primitives.

All sample lists are mono, finite PCM float amplitudes sampled at
``SAMPLE_RATE`` (48 kHz). Bucket widths are counted in integer sample
frames; a bucket is the half-open frame interval
``[i * bucket, (i + 1) * bucket)``. "Peaks" are ``(minimum, maximum)``
amplitude pairs. RMS means ``sqrt(mean(sample ** 2))``. Numeric
comparisons use an absolute tolerance of ``1e-6``.
"""

import math

SAMPLE_RATE = 48_000
TOLERANCE = 1e-6


class WaveformError(ValueError):
    """Raised when waveform input or cache configuration is invalid."""


def _require_positive_int_bucket(bucket) -> int:
    """Return ``bucket`` as a positive int or raise :class:`WaveformError`.

    Fractional widths (e.g. ``2.5``), zero, negatives and booleans are
    rejected so the half-open partition always aligns to whole frames.
    """

    # ``bool`` is an ``int`` subclass but is never a valid bucket width.
    if isinstance(bucket, bool) or not isinstance(bucket, int) or bucket <= 0:
        raise WaveformError(
            f"bucket width must be a positive integer, got {bucket!r}"
        )
    return bucket


def _require_peak_tuple(peak) -> tuple[float, float]:
    """Validate one ``(minimum, maximum)`` record with ``minimum <= maximum``."""

    if not isinstance(peak, (tuple, list)) or len(peak) != 2:
        raise WaveformError(f"peak must be a (minimum, maximum) pair, got {peak!r}")
    minimum, maximum = peak
    if isinstance(minimum, bool) or isinstance(maximum, bool):
        raise WaveformError(f"peak values must be numeric, got {peak!r}")
    if not all(isinstance(value, (int, float)) for value in (minimum, maximum)):
        raise WaveformError(f"peak values must be numeric, got {peak!r}")
    if not (math.isfinite(minimum) and math.isfinite(maximum)):
        raise WaveformError(f"peak values must be finite, got {peak!r}")
    if minimum > maximum:
        raise WaveformError(
            f"peak minimum {minimum!r} must not exceed maximum {maximum!r}"
        )
    return float(minimum), float(maximum)


def bucket_peaks(
    samples: list[float], bucket: int
) -> list[tuple[float, float]]:
    """Build one ``(minimum, maximum)`` pair per fixed-width bucket.

    ``samples`` is a mono PCM float sequence at ``SAMPLE_RATE`` (48 kHz);
    no resampling is performed. ``bucket`` is a positive integer number of
    sample frames. Buckets partition the sample frames into half-open
    intervals ``[i * bucket, (i + 1) * bucket)`` so every frame belongs to
    exactly one bucket, including a shorter trailing bucket. Empty input
    returns ``[]``. A bucket that is not a positive integer (including
    ``0`` or a fractional width such as ``2.5``) raises
    :class:`WaveformError`. Results are compared with absolute tolerance
    ``1e-6``.
    """

    bucket = _require_positive_int_bucket(bucket)
    peaks: list[tuple[float, float]] = []
    for start in range(0, len(samples), bucket):
        frame = samples[start : start + bucket]
        # Keeping both extrema preserves the negative half-wave instead
        # of flattening it to a single maximum.
        peaks.append((min(frame), max(frame)))
    return peaks


def bucket_rms(samples: list[float], bucket: int) -> list[float]:
    """Return the root-mean-square amplitude for each fixed-width bucket.

    ``samples`` is a mono PCM float sequence at ``SAMPLE_RATE`` (48 kHz).
    ``bucket`` is a positive integer frame width and uses the same
    half-open partition, including a shorter trailing bucket, as
    :func:`bucket_peaks`. For each bucket the result is
    ``sqrt(sum(x * x for x in bucket_samples) / count)`` — not the bucket
    peak. Empty input returns ``[]``; a non-positive or fractional bucket
    width raises :class:`WaveformError`. Results are valid to absolute
    tolerance ``1e-6``.
    """

    bucket = _require_positive_int_bucket(bucket)
    result: list[float] = []
    for start in range(0, len(samples), bucket):
        frame = samples[start : start + bucket]
        mean_square = sum(value * value for value in frame) / len(frame)
        result.append(math.sqrt(mean_square))
    return result


def multi_resolution(samples: list[float], levels: list[int]) -> list[dict]:
    """Build a coarse-to-fine multi-resolution peak hierarchy.

    ``samples`` is a mono PCM float sequence at ``SAMPLE_RATE`` (48 kHz).
    ``levels`` is a non-empty list of integer bucket widths measured in
    sample frames, ordered coarse to fine. Each width after the first must
    be exactly half of the previous width (for example ``[8, 4, 2]``).
    Every level uses the half-open bucket partition, retains a shorter
    trailing bucket, and is returned as ``{"bucket": width, "peaks":
    [(minimum, maximum), ...]}`` in the same order as ``levels``. Invalid
    levels, including a zero or non-halving width, raise
    :class:`WaveformError`. Peak values use absolute tolerance ``1e-6``.
    """

    if not isinstance(levels, (list, tuple)) or len(levels) == 0:
        raise WaveformError("levels must be a non-empty list of bucket widths")

    hierarchy: list[dict] = []
    previous = None
    for width in levels:
        width = _require_positive_int_bucket(width)
        if previous is not None and width * 2 != previous:
            raise WaveformError(
                f"level width {width} must be exactly half of {previous}"
            )
        # Every level is built directly from the samples with the same
        # half-open partition, so a partial trailing bucket is always kept.
        hierarchy.append({"bucket": width, "peaks": bucket_peaks(samples, width)})
        previous = width
    return hierarchy


def merge_levels(levels: list[dict]) -> dict:
    """Merge a coarse-to-fine peak hierarchy without losing extrema.

    ``levels`` is a list of dictionaries shaped like the return values of
    :func:`multi_resolution`: ``{"bucket": int, "peaks": [(minimum,
    maximum), ...]}``. It must be ordered coarse to fine, use widths that
    halve exactly between adjacent levels, and cover the same 48 kHz sample
    span. Adjacent buckets are merged; each output bucket is the envelope
    of all corresponding source peaks — the minimum of the minima and the
    maximum of the maxima. Values are never averaged. Return
    ``{"bucket": coarsest_width, "peaks": [(minimum, maximum), ...]}``.
    Malformed or misaligned levels raise :class:`WaveformError`. Peak
    values use absolute tolerance ``1e-6``.
    """

    if not isinstance(levels, (list, tuple)) or len(levels) == 0:
        raise WaveformError("levels must be a non-empty hierarchy of peak levels")

    parsed: list[tuple[int, list[tuple[float, float]]]] = []
    previous_width = None
    for level in levels:
        if not isinstance(level, dict) or "bucket" not in level or "peaks" not in level:
            raise WaveformError(f"malformed peak level: {level!r}")
        width = _require_positive_int_bucket(level["bucket"])
        if previous_width is not None and width * 2 != previous_width:
            raise WaveformError(
                f"level width {width} must be exactly half of {previous_width}"
            )
        raw_peaks = level["peaks"]
        if not isinstance(raw_peaks, (list, tuple)):
            raise WaveformError("level peaks must be a list of (minimum, maximum) pairs")
        parsed.append((width, [_require_peak_tuple(peak) for peak in raw_peaks]))
        previous_width = width

    coarsest_width, coarsest_peaks = parsed[0]
    bucket_count = len(coarsest_peaks)

    # Every finer level must cover the same frame span. All but the last
    # (possibly partial) coarsest bucket hold exactly ``factor`` fine
    # buckets; the trailing one holds between 1 and ``factor``. So a level
    # with ``n`` coarse buckets has ``factor*(n-1)+1 .. factor*n`` buckets.
    for width, peaks in parsed:
        factor = coarsest_width // width
        if bucket_count == 0:
            if len(peaks) != 0:
                raise WaveformError("level bucket counts are misaligned")
        else:
            lower = factor * (bucket_count - 1) + 1
            upper = factor * bucket_count
            if not (lower <= len(peaks) <= upper):
                raise WaveformError(
                    "level bucket counts are misaligned with the coarsest partition"
                )

    merged: list[tuple[float, float]] = []
    for index in range(bucket_count):
        minimum = math.inf
        maximum = -math.inf
        for width, peaks in parsed:
            factor = coarsest_width // width
            for fine_index in range(index * factor, (index + 1) * factor):
                if fine_index >= len(peaks):
                    continue  # partial trailing bucket absent on this level
                low, high = peaks[fine_index]
                minimum = min(minimum, low)
                maximum = max(maximum, high)
        merged.append((minimum, maximum))

    return {"bucket": coarsest_width, "peaks": merged}


def query_range(
    peaks: list[tuple[float, float]],
    start: int,
    end: int,
    bucket: int,
) -> list[tuple]:
    """Return peaks whose buckets start inside a half-open frame range.

    ``peaks`` contains consecutive bucket records for 48 kHz audio.
    ``bucket`` is their positive integer frame width, so bucket ``i`` starts
    at frame ``i * bucket``. ``start`` and ``end`` are non-negative integer
    sample-frame offsets aligned to bucket boundaries, and the query is the
    half-open interval ``[start, end)``. Return one tuple per selected
    bucket as ``(bucket_start, minimum, maximum)``. A bucket starting
    exactly at ``end`` is excluded. Misaligned offsets, invalid bucket
    widths, or ``start > end`` raise :class:`WaveformError`. Amplitude
    values use absolute tolerance ``1e-6``.
    """

    bucket = _require_positive_int_bucket(bucket)
    for name, offset in (("start", start), ("end", end)):
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise WaveformError(f"{name} must be a non-negative integer, got {offset!r}")
        if offset % bucket != 0:
            raise WaveformError(
                f"{name}={offset} is not aligned to bucket width {bucket}"
            )
    if start > end:
        raise WaveformError(f"start {start} must not exceed end {end}")

    first = start // bucket
    last = end // bucket  # exclusive: the bucket at ``end`` is excluded
    selected: list[tuple] = []
    for index in range(first, min(last, len(peaks))):
        minimum, maximum = peaks[index]
        selected.append((index * bucket, minimum, maximum))
    return selected


def incremental_update(
    peaks: list[tuple[float, float]],
    bucket: int,
    added: list[tuple[float, float]],
    removed: int,
) -> list[tuple[float, float]]:
    """Update the tail of a peak cache without recomputing all buckets.

    ``peaks`` is a current list of 48 kHz bucket ``(minimum, maximum)``
    records and ``bucket`` is the positive integer frame width of every
    record. ``removed`` is the non-negative number of oldest whole buckets
    to evict; ``added`` is the list of complete peak tuples appended for
    new trailing buckets. Only the evicted prefix and appended suffix are
    touched — untouched records must not be recomputed; the work is
    proportional to ``removed + len(added)``, not a full O(n²) rebuild.
    Return a new list and leave the input list unchanged. A non-positive
    bucket, a negative or oversized ``removed`` value, or an invalid peak
    tuple raises :class:`WaveformError`. Peak values use absolute tolerance
    ``1e-6``.
    """

    bucket = _require_positive_int_bucket(bucket)
    if not isinstance(peaks, (list, tuple)):
        raise WaveformError("peaks must be a list of (minimum, maximum) records")
    if isinstance(removed, bool) or not isinstance(removed, int) or removed < 0:
        raise WaveformError(f"removed must be a non-negative integer, got {removed!r}")
    if removed > len(peaks):
        raise WaveformError(
            f"cannot remove {removed} buckets from a cache of {len(peaks)}"
        )
    if not isinstance(added, (list, tuple)):
        raise WaveformError("added must be a list of (minimum, maximum) records")

    retained = [_require_peak_tuple(peak) for peak in peaks[removed:]]
    appended = [_require_peak_tuple(peak) for peak in added]
    # Slice + concatenation touches only the evicted prefix and the new
    # suffix; the retained records are never recomputed.
    return retained + appended


def normalize_peaks(
    peaks: list[tuple[float, float]], ceiling: float
) -> list[tuple[float, float]]:
    """Normalize peak extrema by the global peak while preserving silence.

    ``peaks`` contains 48 kHz bucket ``(minimum, maximum)`` records.
    Find the largest absolute amplitude across every minimum and maximum,
    then scale both extrema of every bucket by ``ceiling / global_peak`` so
    the normalized global peak equals ``ceiling``. A fully silent input
    (global peak exactly zero) returns all-zero pairs and must not divide
    by zero or add noise; silent buckets inside a non-silent input stay
    at zero. ``ceiling`` must be positive and finite, otherwise
    :class:`WaveformError` is raised. Results use absolute tolerance
    ``1e-6``.
    """

    if (
        isinstance(ceiling, bool)
        or not isinstance(ceiling, (int, float))
        or not math.isfinite(ceiling)
        or ceiling <= 0
    ):
        raise WaveformError(f"ceiling must be a positive finite number, got {ceiling!r}")

    global_peak = 0.0
    for minimum, maximum in peaks:
        global_peak = max(global_peak, abs(minimum), abs(maximum))

    if global_peak == 0.0:
        # Fully silent track: stay exactly zero, never divide by zero.
        return [(0.0, 0.0) for _ in peaks]

    scale = ceiling / global_peak
    return [(minimum * scale, maximum * scale) for minimum, maximum in peaks]


def peaks_batch(tracks: dict[str, list[float]], config: dict) -> dict:
    """Generate peak caches for multiple tracks atomically.

    ``tracks`` maps track names to mono PCM float sequences at
    ``SAMPLE_RATE`` (48 kHz). ``config`` requires an integer ``"bucket"``
    frame width and may set ``"normalize"`` (bool, default ``False``) and
    ``"ceiling"`` (positive float, default ``1.0``). Return a new dict in
    the same insertion order as ``tracks``, mapping each name to its own
    independent peak list. Tracks must never share a mutable bucket list or
    leak state into one another. Validate every track and the config before
    exposing any result: if any track is invalid or non-finite, raise
    :class:`WaveformError` and roll back the whole batch so no partial
    output is returned. Peak values use absolute tolerance ``1e-6``.
    """

    if not isinstance(config, dict):
        raise WaveformError("config must be a dictionary")
    bucket = _require_positive_int_bucket(config.get("bucket"))

    normalize = config.get("normalize", False)
    if not isinstance(normalize, bool):
        raise WaveformError(f'"normalize" must be a boolean, got {normalize!r}')

    ceiling = config.get("ceiling", 1.0)
    if (
        isinstance(ceiling, bool)
        or not isinstance(ceiling, (int, float))
        or not math.isfinite(ceiling)
        or ceiling <= 0
    ):
        raise WaveformError(
            f'"ceiling" must be a positive finite number, got {ceiling!r}'
        )

    if not isinstance(tracks, dict):
        raise WaveformError("tracks must be a name -> samples mapping")

    # Validate every track before building anything, so a single invalid
    # track rolls back the whole batch with no partial results exposed.
    for name, samples in tracks.items():
        if not isinstance(samples, (list, tuple)):
            raise WaveformError(f"track {name!r} must be a list of PCM samples")
        for value in samples:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise WaveformError(
                    f"track {name!r} contains a non-numeric sample: {value!r}"
                )
            if not math.isfinite(value):
                raise WaveformError(
                    f"track {name!r} contains a non-finite sample: {value!r}"
                )

    result: dict = {}
    for name, samples in tracks.items():
        # Each track gets a freshly built, independent bucket list.
        peaks = bucket_peaks(samples, bucket)
        if normalize:
            peaks = normalize_peaks(peaks, ceiling)
        result[name] = peaks
    return result
