"""Red acceptance tests for the waveform peak cache.

Tests are the evaluation baseline and must not be modified.
"""

import math

import pytest

from waveform import (
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


# 缺陷 1：桶宽用浮点，边界落到两个桶。
def test_float_bucket_width_is_rejected():
    with pytest.raises(WaveformError):
        bucket_peaks([-0.5, 0.5, 0.25], 2.5)


# 缺陷 2：每桶只记 max，波形不对称。
def test_bucket_peaks_preserves_negative_and_positive_extrema():
    peaks = bucket_peaks([-0.8, -0.2, 0.3, 0.7], 2)
    assert peaks == [(-0.8, -0.2), (0.3, 0.7)]


# 缺陷 3：RMS 用峰值代替，能量显示偏高。
def test_bucket_rms_matches_root_mean_square():
    rms = bucket_rms([0.0, 1.0, -1.0, 0.0], 4)
    assert rms == pytest.approx([math.sqrt(0.5)], abs=1e-6)


# 缺陷 4：多分辨率层级桶宽不能整除，末层缺失。
def test_multi_resolution_keeps_partial_tail_buckets():
    samples = [-1.0, 0.25, 0.0, 0.5, 0.75, -0.25, 0.1, 0.2, 0.9]
    levels = multi_resolution(samples, [4, 2])

    assert [level["bucket"] for level in levels] == [4, 2]
    assert len(levels[0]["peaks"]) == 3
    assert levels[0]["peaks"][-1] == (0.9, 0.9)
    assert len(levels[1]["peaks"]) == 5
    assert levels[1]["peaks"][-1] == (0.9, 0.9)


# 缺陷 5：合并层级取平均，峰值被削。
def test_merge_levels_keeps_extrema_not_average():
    levels = [
        {"bucket": 4, "peaks": [(0.0, 0.0)]},
        {"bucket": 2, "peaks": [(-1.0, 0.25), (0.1, 0.8)]},
    ]

    merged = merge_levels(levels)

    assert merged["bucket"] == 4
    assert merged["peaks"] == [(-1.0, 0.8)]


# 缺陷 6：范围查询用 <= 而不是 <，多算一个桶。
def test_query_range_excludes_bucket_at_end():
    peaks = [(-1.0, 0.0), (0.0, 0.4), (-0.3, 0.3), (0.2, 0.9)]

    assert query_range(peaks, 2, 6, 2) == [
        (2, 0.0, 0.4),
        (4, -0.3, 0.3),
    ]


# 缺陷 7：增量更新重算全量，O(n²)。
def test_incremental_update_preserves_untouched_cache_entries():
    current = [(-1.0, -0.5), (0.0, 1.0), (-0.2, 0.2)]

    updated = incremental_update(current, 2, [(0.1, 0.6)], 1)

    assert updated == [current[1], current[2], (0.1, 0.6)]


# 缺陷 8：归一时静音轨除以 0。
def test_normalize_peaks_keeps_silence_silent():
    silent = [(0.0, 0.0), (0.0, 0.0)]

    assert normalize_peaks(silent, 1.0) == [(0.0, 0.0), (0.0, 0.0)]


# 缺陷 9：批量共用桶列表，轨道之间串。
def test_peaks_batch_isolates_track_buffers():
    tracks = {
        "short": [0.2, 0.4, 0.6],
        "long": [-0.9, -0.9, -0.9, -0.9, -0.9],
    }

    result = peaks_batch(tracks, {"bucket": 2})

    assert result["short"] == [(0.2, 0.4), (0.6, 0.6)]
    assert result["long"] == [(-0.9, -0.9)] * 3


# 缺陷 10：bucket=0 不校验，死循环。
def test_multi_resolution_rejects_zero_bucket_level():
    with pytest.raises(WaveformError):
        multi_resolution([0.1, -0.1], [0])


# 组合 1：单桶峰值等于全局 min/max。
def test_single_bucket_equals_global_extrema():
    samples = [-0.75, 0.25, 0.5, -0.1]

    assert bucket_peaks(samples, len(samples)) == [(-0.75, 0.5)]


# 组合 2：bucket=1 时桶数等于样本数。
def test_bucket_width_one_has_one_peak_per_sample():
    samples = [-0.2, 0.3, 0.0]

    assert bucket_peaks(samples, 1) == [
        (-0.2, -0.2),
        (0.3, 0.3),
        (0.0, 0.0),
    ]


# 组合 3：静音轨归一后全零。
def test_silent_batch_track_normalizes_to_zero():
    result = peaks_batch(
        {"silent": [0.0, 0.0, 0.0, 0.0, 0.0]},
        {"bucket": 2, "normalize": True, "ceiling": 1.0},
    )

    assert result["silent"] == [(0.0, 0.0)] * 3


# 组合 4：bucket=0 抛 WaveformError。
def test_bucket_zero_raises_waveform_error():
    with pytest.raises(WaveformError):
        bucket_peaks([0.0, 0.1], 0)


# 组合 5：批量全成功时顺序不变。
def test_batch_success_preserves_track_order():
    tracks = {
        "first": [0.1, 0.2],
        "second": [-0.3, -0.4, 0.5],
        "third": [0.0],
    }

    result = peaks_batch(tracks, {"bucket": 2})

    assert list(result) == list(tracks)
    assert result["first"] == [(0.1, 0.2)]
    assert result["second"] == [(-0.4, -0.3), (0.5, 0.5)]
    assert result["third"] == [(0.0, 0.0)]
