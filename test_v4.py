#!/usr/bin/env python3
"""
Test script v4 - sentence-based alignment strategy.
"""

import re


def align_text_to_timeline(timeline_subs, reference_text, max_chars=20):
    """
    Align reference text to Whisper timeline using sentence-based distribution.

    Strategy:
    1. Split reference text into complete sentences/phrases by punctuation
    2. Distribute these complete segments across the timeline based on duration
    3. Each time segment gets complete sentences, not fragments
    4. Merge short segments to adjacent ones if needed
    """
    ref_text = reference_text.strip()
    if not ref_text:
        return timeline_subs

    # Remove bracket tags
    ref_text_no_tags = re.sub(r"\[[^\]]*\]", "", ref_text)

    # Normalize whitespace but preserve sentence structure
    ref_clean = ref_text_no_tags.replace("\n", " ").replace("\t", " ")
    ref_clean = re.sub(r"\s+", " ", ref_clean).strip()

    if not timeline_subs:
        return []

    # Calculate total audio duration
    total_duration = timeline_subs[-1]["end"] - timeline_subs[0]["start"]

    # Step 1: Split reference text into semantic units (phrases/sentences)
    semantic_chunks = re.split(r"([，。！？、；：])", ref_clean)

    # Re-attach punctuation to create complete semantic units
    ref_segments = []
    i = 0
    while i < len(semantic_chunks):
        if i + 1 < len(semantic_chunks) and semantic_chunks[i + 1] in "，。！？、；：":
            combined = (semantic_chunks[i] + semantic_chunks[i + 1]).strip()
            if combined:
                ref_segments.append(combined)
            i += 2
        else:
            if semantic_chunks[i].strip():
                ref_segments.append(semantic_chunks[i].strip())
            i += 1

    # Step 2: Assign semantic segments to timeline
    ref_len = len(ref_clean)
    chars_per_second = ref_len / total_duration if total_duration > 0 else 10

    aligned_subs = []
    segment_idx = 0  # Current position in ref_segments

    # Filter out very short timeline segments
    valid_segments = [s for s in timeline_subs if s["end"] - s["start"] >= 0.3]

    # Calculate total duration of valid segments
    valid_duration = sum(s["end"] - s["start"] for s in valid_segments)
    if valid_duration == 0:
        valid_duration = total_duration

    # Calculate how many semantic segments each time period should get
    segments_count = len(ref_segments)
    segments_per_second = segments_count / valid_duration if valid_duration > 0 else 1

    for time_seg in valid_segments:
        duration = time_seg["end"] - time_seg["start"]

        # Calculate how many semantic segments this time period should get
        expected_segments = max(1, int(duration * segments_per_second))

        # Collect semantic segments for this period
        segment_text = ""
        segments_added = 0

        # Add complete semantic segments
        while (
            segment_idx < len(ref_segments)
            and segments_added < expected_segments
            and len(segment_text) + len(ref_segments[segment_idx]) + 1 <= max_chars
        ):
            if segment_text:
                segment_text += " " + ref_segments[segment_idx]
            else:
                segment_text = ref_segments[segment_idx]
            segments_added += 1
            segment_idx += 1

        # Clean up
        segment_text = segment_text.strip()

        # Skip if empty
        if not segment_text:
            continue

        # If segment exceeds max_chars, split it (should be rare)
        if len(segment_text) > max_chars:
            # Split by character count
            aligned_subs.append(
                {
                    "start": time_seg["start"],
                    "end": time_seg["end"],
                    "text": segment_text[:max_chars].strip(),
                }
            )
        else:
            # Single subtitle
            aligned_subs.append(
                {
                    "start": time_seg["start"],
                    "end": time_seg["end"],
                    "text": segment_text,
                }
            )

    # Step 3: Handle remaining text (if any)
    if segment_idx < len(ref_segments):
        remaining_text = " ".join(ref_segments[segment_idx:])

        # Try to add to last subtitle
        if (
            aligned_subs
            and len(aligned_subs[-1]["text"]) + len(remaining_text) + 1 <= max_chars
        ):
            aligned_subs[-1]["text"] += " " + remaining_text
        else:
            # Create new subtitles for remaining text
            if aligned_subs:
                last_end = aligned_subs[-1]["end"]
            else:
                last_end = 0

            chunks = []
            for j in range(0, len(remaining_text), max_chars):
                chunk = remaining_text[j : j + max_chars].strip()
                if chunk:
                    chunks.append(chunk)

            chunk_duration = 2.0
            for chunk in chunks:
                aligned_subs.append(
                    {"start": last_end, "end": last_end + chunk_duration, "text": chunk}
                )
                last_end += chunk_duration

    # Step 4: Post-processing: Merge very short subtitles
    i = 1
    while i < len(aligned_subs):
        current = aligned_subs[i]
        prev = aligned_subs[i - 1]

        # Merge very short (< 4 chars) into previous
        if len(current["text"]) < 4 and i > 0:
            combined = prev["text"] + " " + current["text"]
            if len(combined) <= max_chars:
                prev["text"] = combined
                prev["end"] = current["end"]
                aligned_subs.pop(i)
                continue

        i += 1

    # Ensure we didn't miss any text due to skipped short segments
    final_text = " ".join(sub["text"] for sub in aligned_subs)
    if len(final_text) < len(ref_clean) * 0.95:  # If we missed more than 5%
        # Fallback: append missing text
        if aligned_subs:
            aligned_subs.append(
                {
                    "start": aligned_subs[-1]["end"],
                    "end": aligned_subs[-1]["end"] + 2.0,
                    "text": ref_clean[len(final_text) :].strip(),
                }
            )

    return aligned_subs


# Mock Whisper timeline subs (simulating what Whisper would return)
MOCK_TIMELINE_SUBS = [
    {"start": 0.0, "end": 5.0, "text": "看着k线图心潮澎湃 梦想着别墅还有那跑车"},
    {"start": 5.0, "end": 10.0, "text": "开 专家说牛市马上就到来"},
    {"start": 10.0, "end": 13.5, "text": "梭哈是一种过人的气概"},
    {"start": 13.5, "end": 17.0, "text": "心跳在加速 呼吸都停摆"},
    {"start": 17.0, "end": 24.0, "text": "好像全世界为我喝彩"},
    {"start": 24.0, "end": 29.0, "text": "我要赚得更多 就像石头一样坚硬执着"},
    {"start": 29.0, "end": 36.0, "text": "我要赚得更多 就算账户的绿色吞噬了我"},
    {"start": 36.0, "end": 42.0, "text": "我要赚得更多 在这涨跌的游戏里放肆地活"},
    {"start": 42.0, "end": 49.0, "text": "没有什么能够阻挡 我对财富的向往"},
    {"start": 49.0, "end": 52.5, "text": "才过了几天风向就更改"},
    {"start": 52.5, "end": 56.0, "text": "屏幕的绿色深得像片海 身边的兄弟都成了韭菜"},
    {"start": 56.0, "end": 60.0, "text": "天台的风儿吹得我好呆"},
    {"start": 60.0, "end": 64.0, "text": "眼泪流下来 没人会理睬"},
    {"start": 64.0, "end": 68.0, "text": "原来我只是小小的尘埃"},
    {"start": 68.0, "end": 74.0, "text": "我要赚得更多 就像石头一样坚硬执着"},
    {"start": 74.0, "end": 80.0, "text": "我要赚得更多 就算账户的绿色吞噬了我"},
    {"start": 80.0, "end": 86.0, "text": "我要赚得更多 在这涨跌的游戏里放肆地活"},
    {"start": 86.0, "end": 92.0, "text": "没有什么能够阻挡 我对财富的向往"},
    {"start": 92.0, "end": 95.0, "text": "他们说这是价值的投资"},
    {"start": 95.0, "end": 98.0, "text": "他们说别怕只是技术调整"},
    {
        "start": 98.0,
        "end": 102.0,
        "text": "可我的本金已渐渐消失 只剩下账户里无尽的讽刺",
    },
    {"start": 102.0, "end": 108.0, "text": "我要赚得更多 就像石头一样坚硬执着"},
    {"start": 108.0, "end": 114.0, "text": "我要赚得更多 就算账户的绿色吞噬了我"},
    {"start": 114.0, "end": 120.0, "text": "我要赚得更多 在这涨跌的游戏里放肆地活"},
    {"start": 120.0, "end": 126.0, "text": "没有什么能够阻挡 我对财富的向往"},
    {"start": 126.0, "end": 131.0, "text": "赚得更多 明天会红的吧"},
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

[Verse]
才过了几天风向就更改
屏幕的绿色深得像片海
身边的兄弟都成了韭菜
天台的风儿吹得我好呆

[Pre-Chorus]
眼泪流下来 没人会理睬
原来我只是小小的尘埃

[Chorus]
我要赚得更多 就像石头一样坚硬执着
我要赚得更多 就算账户的绿色吞噬了我
我要赚得更多 在这涨跌的游戏里放肆地活
没有什么能够阻挡 我对财富的向往

[Guitar Solo]

[Bridge]
他们说这是价值的投资
他们说别怕只是技术调整
可我的本金已渐渐消失
只剩下账户里无尽的讽刺

[Chorus]
我要赚得更多 就像石头一样坚硬执着
我要赚得更多 就算账户的绿色吞噬了我
我要赚得更多 在这涨跌的游戏里放肆地活
没有什么能够阻挡 我对财富的向往

[Outro]
赚得更多
明天会红的吧
"""


def test_alignment():
    """Test alignment function."""

    print("=" * 80)
    print("Testing Sentence-Based Alignment v4")
    print("=" * 80)

    # Run alignment
    aligned_subs = align_text_to_timeline(
        MOCK_TIMELINE_SUBS, REFERENCE_TEXT, max_chars=20
    )

    print(f"\nGenerated {len(aligned_subs)} subtitle segments:")
    print("=" * 80)

    # Display results
    for i, sub in enumerate(aligned_subs):
        print(
            f"\n{i + 1}. [{sub['start']:.3f}s -> {sub['end']:.3f}s] ({len(sub['text'])} chars)"
        )
        print(f"   Text: {sub['text']}")

    # Check for issues
    print("\n" + "=" * 80)
    print("Validation Checks:")
    print("=" * 80)

    # Remove bracket tags from reference for comparison
    ref_clean = re.sub(r"\[[^\]]*\]", "", REFERENCE_TEXT)
    ref_clean = re.sub(r"\s+", " ", ref_clean).strip()

    all_aligned_text = " ".join(sub["text"] for sub in aligned_subs)

    print(f"\n1. Text Coverage:")
    print(f"   Reference text: {len(ref_clean)} chars")
    print(f"   Aligned text:  {len(all_aligned_text)} chars")
    print(f"   Coverage: {len(all_aligned_text) / len(ref_clean) * 100:.1f}%")

    # Check for semantic coherence
    print(f"\n2. Semantic Coherence Check:")
    cross_sentence_issues = []
    for i, sub in enumerate(aligned_subs):
        text = sub["text"]
        if "到来 梭哈" in text:
            cross_sentence_issues.append(f"   Sub {i + 1}: '到来 梭哈' (跨句)")
        if "气概 心跳" in text:
            cross_sentence_issues.append(f"   Sub {i + 1}: '气概 心跳' (跨句)")
        if "停摆 我要" in text:
            cross_sentence_issues.append(f"   Sub {i + 1}: '停摆 我要' (跨句)")
        if "喝彩 我要" in text:
            cross_sentence_issues.append(f"   Sub {i + 1}: '喝彩 我要' (跨句)")

    if cross_sentence_issues:
        print("   ❌ FOUND CROSS-SENTENCE CUTS:")
        for issue in cross_sentence_issues:
            print(f"     {issue}")
    else:
        print("   ✓ No obvious cross-sentence cuts detected")

    # Check for complete sentences
    print(f"\n3. Complete Sentence Check:")
    complete_sentences = sum(
        1 for sub in aligned_subs if sub["text"][-1] in "。！？；："
    )
    print(
        f"   Complete sentences (ending with 。！？；：): {complete_sentences}/{len(aligned_subs)}"
    )

    # Display first 20 subtitles
    print("\n" + "=" * 80)
    print("First 15 Subtitles:")
    print("=" * 80)
    for i, sub in enumerate(aligned_subs[:15]):
        print(f"\n{i + 1}. {sub['text']}")

    # Generate SRT
    srt_output = ""
    for i, sub in enumerate(aligned_subs):
        s = sub["start"]
        e = sub["end"]
        txt = sub["text"]

        s_h, s_r = divmod(s, 3600)
        s_m, s_s = divmod(s_r, 60)
        s_ms = (s_s - int(s_s)) * 1000

        e_h, e_r = divmod(e, 3600)
        e_m, e_s = divmod(e_r, 60)
        e_ms = (e_s - int(e_s)) * 1000

        srt_output += f"{i + 1}\n{int(s_h):02}:{int(s_m):02}:{int(s_s):02},{int(s_ms):03} --> {int(e_h):02}:{int(e_m):02}:{int(e_s):02},{int(e_ms):03}\n{txt}\n\n"

    with open("test_output_v4.srt", "w", encoding="utf-8") as f:
        f.write(srt_output)
    print(f"\n✓ SRT saved to test_output_v4.srt")


if __name__ == "__main__":
    test_alignment()
