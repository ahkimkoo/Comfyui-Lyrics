#!/usr/bin/env python3
"""
Test semantic unit distribution (方案A).
"""

import re


def align_text_to_timeline(timeline_subs, reference_text, max_chars=20):
    """Test implementation of 方案A."""
    ref_text = reference_text.strip()
    if not ref_text:
        return timeline_subs

    # Remove bracket tags
    ref_text_no_tags = re.sub(r"\[[^\]]*\]", "", ref_text)

    # Normalize whitespace to single spaces
    ref_clean = ref_text_no_tags.replace("\n", " ").replace("\t", " ")
    ref_clean = re.sub(r"\s+", " ", ref_clean).strip()

    if not timeline_subs:
        return []

    # Split reference text into semantic units
    # Split by Chinese punctuation
    semantic_chunks = re.split(r"([，。！？、；：])", ref_clean)

    # Re-attach punctuation
    semantic_units = []
    i = 0
    while i < len(semantic_chunks):
        if i + 1 < len(semantic_chunks) and semantic_chunks[i + 1] in "，。！？、；：":
            combined = (semantic_chunks[i] + semantic_chunks[i + 1]).strip()
            if combined:
                semantic_units.append(combined)
            i += 2
        else:
            if semantic_chunks[i].strip():
                semantic_units.append(semantic_chunks[i].strip())
                i += 1

    # Distribute across timeline
    num_timeline_segments = len(timeline_subs)
    num_units = len(semantic_units)

    aligned_subs = []
    unit_idx = 0

    for time_seg in timeline_subs:
        duration = time_seg["end"] - time_seg["start"]

        if duration <= 0:
            continue

        # Calculate units for this segment
        target_units = max(1, int((unit_idx + 1) / num_timeline_segments) - unit_idx)

        # Collect units
        units = []
        segment_text = ""

        # Use while loop to collect units until target reached
        collected = 0
        while collected < target_units and unit_idx < num_units:
            segment_text += semantic_units[unit_idx] + 1] + " "
            unit_idx += 1
            collected += 1

        segment_text = segment_text.strip()

        if not segment_text:
            continue

        # Check if exceeds max_chars
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

        else:
            aligned_subs.append({
                "start": time_seg["start"],
                "end": time_seg["end"],
                "text": segment_text
            })

    return aligned_subs


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
    {"start": 114.0, "end": 120.0, "end": 126.0, "text": "ignored"},
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
我要赚得更多 就算账户的绿色吞噬了我
我要赚得更多 在这涨跌的游戏里放肆地活
没有什么能够阻挡 我对财富的向往
"""

if __name__ == "__main__":
    aligned = align_text_to_timeline(MOCK_TIMELINE_SUBS, REFERENCE_TEXT, 20)

    print(f"Generated {len(aligned)} subtitles:\n")
    for i, sub in enumerate(aligned):
        print(
            f"{i + 1}. [{sub['start']:.1f}s -> {sub['end']:.1f}s] ({len(sub['text'])} chars)"
        )
        print(f"   {sub['text']}")

    # Check for cross-sentence issues
    print("\n\nChecking for cross-sentence cuts:")
    issues = []
    for i, sub in enumerate(aligned):
        text = sub["text"]
        if "到来 梭哈" in text:
            issues.append(f"Sub {i + 1}: '到来 梭哈'")
        if "气概 心跳" in text:
            issues.append(f"Sub {i + 1}: '气概 心跳'")
        if "停摆 我要" in text:
            issues.append(f"Sub {i + 1}: '停摆 我要'")
        if "喝彩 我要" in text:
            issues.append(f"Sub {i + 1}: '喝彩 我要'")

    if issues:
        print("❌ FOUND CROSS-SENTENCE CUTS:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✓ No obvious cross-sentence cuts detected")

    # Check semantic completeness
    complete = sum(1 for sub in aligned if sub["text"][-1] in "，。！？；：")
    print(
        f"\nSemantic completeness: {complete}/{len(aligned)} subtitles end with punctuation"
    )
