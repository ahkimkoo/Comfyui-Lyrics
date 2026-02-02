#!/usr/bin/env python3
"""
Final simple alignment test v5.
"""

import re


def align_text_to_timeline(timeline_subs, reference_text, max_chars=20):
    """
    Simple time-based slicing algorithm.
    """
    ref_text = reference_text.strip()
    if not ref_text:
        return timeline_subs

    # Remove bracket tags
    ref_text_no_tags = re.sub(r"\[[^\]]*\]", "", ref_text)

    # Normalize whitespace
    ref_clean = ref_text_no_tags.replace("\n", " ").replace("\t", " ")
    ref_clean = re.sub(r"\s+", " ", ref_clean).strip()

    if not timeline_subs:
        return []

    # Calculate total audio duration
    total_duration = timeline_subs[-1]["end"] - timeline_subs[0]["start"]
    ref_len = len(ref_clean)
    chars_per_second = ref_len / total_duration if total_duration > 0 else 10

    aligned_subs = []
    current_pos = 0  # Current position in ref_clean

    for time_seg in timeline_subs:
        duration = time_seg["end"] - time_seg["start"]

        # Skip invalid segments
        if duration <= 0:
            continue

        # Calculate end position for this segment
        target_end = int(current_pos + duration * chars_per_second)
        target_end = min(target_end, ref_len)

        # Slice text for this segment
        segment_text = ref_clean[current_pos:target_end].strip()

        # If segment is too long, split it
        if len(segment_text) > max_chars:
            # Split at max_chars
            num_chunks = int(len(segment_text) / max_chars) + 1
            chunk_duration = duration / num_chunks

            for i in range(num_chunks):
                start_char = i * max_chars
                end_char = min(start_char + max_chars, len(segment_text))
                chunk_text = segment_text[start_char:end_char].strip()

                if chunk_text:
                    aligned_subs.append(
                        {
                            "start": time_seg["start"] + i * chunk_duration,
                            "end": min(
                                time_seg["start"] + (i + 1) * chunk_duration,
                                time_seg["end"],
                            ),
                            "text": chunk_text,
                        }
                    )

            current_pos = target_end
        else:
            # Single subtitle for this segment
            aligned_subs.append(
                {
                    "start": time_seg["start"],
                    "end": time_seg["end"],
                    "text": segment_text,
                }
            )
            current_pos = target_end

    # Post-process: merge very short (<3 chars) subtitles
    i = 1
    while i < len(aligned_subs):
        current = aligned_subs[i]

        # If very short and has next, try to merge
        if len(current["text"]) < 3 and i + 1 < len(aligned_subs):
            next_sub = aligned_subs[i + 1]
            combined = current["text"] + " " + next_sub["text"]

            if len(combined) <= max_chars:
                # Merge into current
                current["text"] = combined
                current["end"] = next_sub["end"]
                aligned_subs.pop(i + 1)
                continue

        i += 1

    return aligned_subs


# Test data
MOCK_TIMELINE_SUBS = [
    {"start": 0.0, "end": 5.0, "text": "text1"},
    {"start": 5.0, "end": 10.0, "text": "text2"},
    {"start": 10.0, "end": 13.5, "text": "text3"},
    {"start": 13.5, "end": 17.0, "text": "text4"},
    {"start": 17.0, "end": 24.0, "text": "text5"},
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

[Chorus]
我要赚得更多 就像石头一样坚硬执着
我要赚得更多 就算账户的绿色吞噬了我
我要赚得更多 在这涨跌的游戏里放肆地活
没有什么能够阻挡 我对财富的向往

[Outro]
赚得更多
明天会红的吧
"""


def test():
    aligned = align_text_to_timeline(MOCK_TIMELINE_SUBS, REFERENCE_TEXT, 20)

    print(f"Generated {len(aligned)} subtitles:")
    for i, sub in enumerate(aligned):
        print(
            f"{i + 1}. [{sub['start']:.1f}s -> {sub['end']:.1f}s] ({len(sub['text'])} chars)"
        )
        print(f"   {sub['text']}")


if __name__ == "__main__":
    test()
