#!/usr/bin/env python3
"""
Final simple test v7 - test the rewritten simple algorithm.
"""

import re

def align_text_to_timeline(timeline_subs, reference_text, max_chars=20):
    """
    Very simple proportional distribution.
    """
    ref_text = reference_text.strip()
    if not ref_text:
        return timeline_subs

    # Remove bracket tags
    ref_text_no_tags = re.sub(r"\[[^\]]*\]", "", ref_text)

    # Normalize whitespace BUT keep punctuation
    ref_clean = ref_text_no_tags.replace("\n", " ").replace("\t", " ")
    ref_clean = re.sub(r"\s+", " ", ref_clean).strip()

    if not timeline_subs:
        return []

    # Calculate total duration
    total_duration = timeline_subs[-1]["end"] - timeline_subs[0]["start"]

        # Calculate reference text length
        ref_len = len(ref_clean)

        # Characters per second
        chars_per_second = ref_len / total_duration if total_duration > 0 else 10

        aligned_subs = []
        text_position = 0

        # Process each timeline segment
        for time_seg in timeline_subs:
            duration = time_seg["end"] - time_seg["start"]

            # Skip invalid segments
            if duration <= 0:
                continue

            # Calculate target characters for this time segment
            target_chars = int(duration * chars_per_second)

            # Determine end position
            end_position = min(text_position + target_chars, ref_len)

            # Extract text
            segment_text = ref_clean[text_position:end_position].strip()

            # Skip if empty
            if not segment_text:
                continue

            # If exceeds max_chars, split it
            if len(segment_text) > max_chars:
                num_chunks = int(len(segment_text) / max_chars) + 1
                chunk_duration = duration / num_chunks

                for i in range(num_chunks):
                    chunk_start = i * max_chars
                    chunk_end = min(chunk_start + max_chars, len(segment_text))
                    chunk_text = segment_text[chunk_start:chunk_end].strip()

                    if chunk_text:
                        aligned_subs.append({
                            "start": time_seg["start"] + i * chunk_duration,
                            "end": min(time_seg["start"] + (i + 1) * chunk_duration, time_seg["end"]),
                            "text": chunk_text
                        })

                text_position = end_position
            else:
                aligned_subs.append({
                    "start": time_seg["start"],
                    "end": time_seg["end"],
                    "text": segment_text
                })
                text_position = end_position

        # Handle remaining text
        if text_position < ref_len:
            remaining_text = ref_clean[text_position:].strip()

            # Try to append to last subtitle
            if aligned_subs and len(aligned_subs[-1]["text"]) + len(remaining_text) + 1 <= max_chars:
                aligned_subs[-1]["text"] += " " + remaining_text
            else:
                # Create new subtitles
                if aligned_subs:
                    last_end = aligned_subs[-1]["end"]
                else:
                    last_end = 0

                # Split remaining text
                chunks = []
                for i in range(0, len(remaining_text), max_chars):
                    chunk = remaining_text[i:i + max_chars].strip()
                    if chunk:
                        chunks.append(chunk)

                chunk_duration = 2.0
                for chunk in chunks:
                    aligned_subs.append({
                        "start": last_end,
                        "end": last_end + chunk_duration,
                        "text": chunk
                    })
                    last_end += chunk_duration

        # Post-process: merge very short (< 3 chars) subtitles
        i = 1
        while i < len(aligned_subs):
            current = aligned_subs[i]
            prev = aligned_subs[i - 1]

            if len(current["text"]) < 3:
                combined = prev["text"] + " " + current["text"]
                if len(combined) <= max_chars:
                    prev["text"] = combined
                    prev["end"] = current["end"]
                    aligned_subs.pop(i)
                    continue

            i += 1

        return aligned_subs


# Test data
MOCK_TIMELINE_SUBS = [
    {"start": 0.0, "end": 5.0, "text": "ignored"},
    {"start": 5.0, "end": 10.0, "text": "ignored"},
    {"start": 10.0, "end": 13.5, "text": "ignored"},
    {"start": 13.5, "end": 17.0, "text": "ignored"},
    {"start": 17.0, "end": 24.0, "text": "ignored"},
    {"start": 24.0, "end": 29.0, "text": "ignored"},
    {"start": 29.0, "end": 36.0, "text": "ignored"},
    {"start": 36.0, "end": 42.0, "text": "ignored"},
    {"start": 42.0, "end": 49.0, "text": "ignored"},
    {"start": 49.0, "end": 52.5, "text": "ignored"},
    {"start": 52.5, "end": 56.0, "text": "ignored"},
    {"start": 56.0, "end": 60.0, "text": "ignored"},
    {"start": 60.0, "end": 64.0, "text": "ignored"},
    {"start": 64.0, "end": 68.0, "text": "ignored"},
    {"start": 68.0, "end": 74.0, "text": "ignored"},
    {"start": 74.0, "end": 80.0, "text": "ignored"},
    {"start": 80.0, "end": 86.0, "text": "ignored"},
    {"start": 86.0, "end": 92.0, "text": "ignored"},
    {"start": 92.0, "end": 95.0, "text": "ignored"},
    {"start": 95.0, "end": 98.0, "text": "ignored"},
    {"start": 98.0, "end": 102.0, "text": "ignored"},
    {"start": 102.0, "end": 108.0, "text": "ignored"},
    {"start": 108.0, "end": 114.0, "text": "ignored"},
    {"start": 114.0, "end": 120.0, "text": "ignored"},
    {"start": 120.0, "end": 126.0, "text": "ignored"},
    {"start": 126.0, "end": 131.0, "text": "ignored"},
]

REFERENCE_TEXT = """
[Verse]
看着k线图心潮澎湃
梦想着别墅还有那跑车开
专家说牛市马上就到来
梭哈是一种过人的气概

[Pre-Chorus]
心跳在加速 呼吸都停摆
好像全世界为我喝彩

[Chorus]
我要赚得更多 就像石头一样坚硬执着
"""

if __name__ == "__main__":
    aligned = align_text_to_timeline(MOCK_TIMELINE_SUBS, REFERENCE_TEXT, 20)

    print(f"Generated {len(aligned)} subtitles:\n")

    # Show first 20
    for i, sub in enumerate(aligned):
        print(f"{i+1}. [{sub['start']:.1f}s -> {sub['end']:.1f}s] ({len(sub['text'])} chars)")
        print(f"   {sub['text']}")
