#!/usr/bin/env python3
"""
Final test v6 - simple proportional distribution.
"""

import re


def align_text_to_timeline(timeline_subs, reference_text, max_chars=20):
    """
    Simple proportional distribution.
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

    # Calculate total duration
    total_duration = timeline_subs[-1]["end"] - timeline_subs[0]["start"]
    ref_len = len(ref_clean)
    chars_per_second = ref_len / total_duration if total_duration > 0 else 10

    aligned_subs = []
    text_position = 0

    for time_seg in timeline_subs:
        duration = time_seg["end"] - time_seg["start"]

        if duration <= 0:
            continue

        # Calculate target chars
        target_chars = int(duration * chars_per_second)

        # Determine end position
        end_position = min(text_position + target_chars, ref_len)

        # Extract segment
        segment_text = ref_clean[text_position:end_position].strip()

        if not segment_text:
            continue

        # Split if too long
        if len(segment_text) > max_chars:
            num_chunks = int(len(segment_text) / max_chars) + 1
            chunk_duration = duration / num_chunks

            for i in range(num_chunks):
                chunk_start = i * max_chars
                chunk_end = min(chunk_start + max_chars, len(segment_text))
                chunk_text = segment_text[chunk_start:chunk_end].strip()

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

            text_position = end_position
        else:
            aligned_subs.append(
                {
                    "start": time_seg["start"],
                    "end": time_seg["end"],
                    "text": segment_text,
                }
            )
            text_position = end_position

    # Merge very short (<4 chars) consecutive subtitles
    i = 1
    while i < len(aligned_subs):
        current = aligned_subs[i]
        prev = aligned_subs[i - 1]

        if len(current["text"]) < 4:
            combined = prev["text"] + " " + current["text"]
            if len(combined) <= max_chars:
                prev["text"] = combined
                prev["end"] = current["end"]
                aligned_subs.pop(i)
                continue

        i += 1

    return aligned_subs


# Test with real data structure
MOCK_TIMELINE_SUBS = [
    {"start": 0.0, "end": 5.0, "text": "t1"},
    {"start": 5.0, "end": 10.0, "text": "t2"},
    {"start": 10.0, "end": 13.5, "text": "t3"},
    {"start": 13.5, "end": 17.0, "text": "t4"},
    {"start": 17.0, "end": 24.0, "text": "t5"},
    {"start": 24.0, "end": 29.0, "text": "t6"},
    {"start": 29.0, "end": 36.0, "text": "t7"},
    {"start": 36.0, "end": 42.0, "text": "t8"},
    {"start": 42.0, "end": 49.0, "text": "t9"},
    {"start": 49.0, "end": 52.5, "text": "t10"},
    {"start": 52.5, "end": 56.0, "text": "t11"},
    {"start": 56.0, "end": 60.0, "text": "t12"},
    {"start": 60.0, "end": 64.0, "text": "t13"},
    {"start": 64.0, "end": 68.0, "text": "t14"},
    {"start": 68.0, "end": 74.0, "text": "t15"},
    {"start": 74.0, "end": 80.0, "text": "t16"},
    {"start": 80.0, "end": 86.0, "text": "t17"},
    {"start": 86.0, "end": 92.0, "text": "t18"},
    {"start": 92.0, "end": 95.0, "text": "t19"},
    {"start": 95.0, "end": 98.0, "text": "t20"},
    {"start": 98.0, "end": 102.0, "text": "t21"},
    {"start": 102.0, "end": 108.0, "text": "t22"},
    {"start": 108.0, "end": 114.0, "text": "t23"},
    {"start": 114.0, "end": 120.0, "text": "t24"},
    {"start": 120.0, "end": 126.0, "text": "t25"},
    {"start": 126.0, "end": 131.0, "text": "t26"},
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

    print(f"Generated {len(aligned)} subtitles:\n")

    # Show first 15
    for i, sub in enumerate(aligned):
        print(
            f"{i + 1}. [{sub['start']:.1f}s -> {sub['end']:.1f}s] ({len(sub['text'])} chars)"
        )
        print(f"   {sub['text']}")

    # Check for cross-sentence issues
    print("\nChecking for cross-sentence cuts:")
    issues = []
    for i, sub in enumerate(aligned):
        text = sub["text"]
        if "到来 梭哈" in text:
            issues.append(f"Sub {i + 1}: '到来 梭哈'")
        if "气概 心跳" in text:
            issues.append(f"Sub {i + 1}: '气概 心跳'")
        if "停摆 我要" in text:
            issues.append(f"Sub {i + 1}: '停摆 我要'")

    if issues:
        print("❌ Found issues:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✓ No obvious cross-sentence cuts")


if __name__ == "__main__":
    test()
