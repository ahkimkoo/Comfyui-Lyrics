#!/usr/bin/env python3
"""
Debug test to understand the algorithm flow.
"""

import re


def align_text_to_timeline(timeline_subs, reference_text, max_chars=20):
    """Test version with debug output."""
    ref_text = reference_text.strip()
    if not ref_text:
        return timeline_subs

    ref_text_no_tags = re.sub(r"\[[^\]]*\]", "", ref_text)
    ref_clean = ref_text_no_tags.replace("\n", " ").replace("\t", " ")
    ref_clean = re.sub(r"\s+", " ", ref_clean).strip()

    print(f"Reference clean text: {ref_clean}")
    print(f"Reference length: {len(ref_clean)}")

    if not timeline_subs:
        return []

    total_duration = timeline_subs[-1]["end"] - timeline_subs[0]["start"]
    print(f"Total duration: {total_duration}s")

    ref_len = len(ref_clean)
    chars_per_second = ref_len / total_duration if total_duration > 0 else 10
    print(f"Chars per second: {chars_per_second:.2f}")

    aligned_subs = []
    text_position = 0

    for i, time_seg in enumerate(timeline_subs[:10]):  # Only first 10
        duration = time_seg["end"] - time_seg["start"]

        if duration <= 0:
            continue

        target_chars = int(duration * chars_per_second)
        end_position = min(text_position + target_chars, ref_len)

        print(f"\n--- Segment {i + 1} ---")
        print(
            f"Time: {time_seg['start']:.1f}s -> {time_seg['end']:.1f}s ({duration:.1f}s)"
        )
        print(f"Text position: {text_position} -> {end_position}")
        print(f"Target chars: {target_chars}")

        segment_text = ref_clean[text_position:end_position].strip()
        print(f"Segment text: '{segment_text}' ({len(segment_text)} chars)")

        if len(segment_text) > max_chars:
            num_chunks = int(len(segment_text) / max_chars) + 1
            chunk_duration = duration / num_chunks
            print(f"Splitting into {num_chunks} chunks")

            for j in range(num_chunks):
                chunk_start = j * max_chars
                chunk_end = min(chunk_start + max_chars, len(segment_text))
                chunk_text = segment_text[chunk_start:chunk_end].strip()

                print(f"  Chunk {j + 1}: '{chunk_text}' ({len(chunk_text)} chars)")

                if chunk_text:
                    aligned_subs.append(
                        {
                            "start": time_seg["start"] + j * chunk_duration,
                            "end": min(
                                time_seg["start"] + (j + 1) * chunk_duration,
                                time_seg["end"],
                            ),
                            "text": chunk_text,
                        }
                    )
            text_position = end_position
        else:
            print(f"Single subtitle: '{segment_text}'")
            aligned_subs.append(
                {
                    "start": time_seg["start"],
                    "end": time_seg["end"],
                    "text": segment_text,
                }
            )
            text_position = end_position

    print(f"\nFinal text_position: {text_position}")

    return aligned_subs


# Simplified test data
MOCK_TIMELINE_SUBS = [
    {"start": 0.0, "end": 5.0, "text": "ignored"},
    {"start": 5.0, "end": 10.0, "text": "ignored"},
    {"start": 10.0, "end": 13.5, "text": "ignored"},
    {"start": 13.5, "end": 17.0, "text": "ignored"},
    {"start": 17.0, "end": 24.0, "text": "ignored"},
]

REFERENCE_TEXT = """
[Intro]

[Verse]
看着k线图心潮澎湃
梦想着别墅还有那跑车开
专家说牛市马上就到来
梭哈是一种过人的气概

[Pre-Chorus]
心跳在加速 呼吸都停摆
好像全世界为我喝彩
"""

if __name__ == "__main__":
    result = align_text_to_timeline(MOCK_TIMELINE_SUBS, REFERENCE_TEXT, 20)
    print(f"\n\nFinal result: {len(result)} subtitles")
