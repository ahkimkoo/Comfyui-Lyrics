#!/usr/bin/env python3
"""
Standalone test script for reference text alignment fix v2.
Tests semantic-aware alignment logic without ComfyUI dependencies.
"""

import re


# Updated semantic-aware version of alignment function for testing
def align_text_to_timeline(timeline_subs, reference_text, max_chars=20):
    """
    Align reference text to Whisper timeline using semantic-aware distribution.

    Key improvements:
    1. Split reference text by semantic boundaries (punctuation, spaces)
    2. Assign complete semantic segments to time periods
    3. Merge short segments and split long ones intelligently
    4. Preserve sentence/phrase coherence
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

    # Step 1: Split reference text by semantic boundaries
    # Keep punctuation as separate segments for better control
    semantic_chunks = re.split(r"([，。！？、；：])", ref_clean)

    # Re-attach punctuation to the preceding text
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

    # Step 2: Calculate time distribution
    ref_len = len(ref_clean)
    chars_per_second = ref_len / total_duration if total_duration > 0 else 10

    aligned_subs = []
    segment_idx = 0  # Current position in ref_segments
    current_time = timeline_subs[0]["start"]

    # Process each time segment
    for i, time_seg in enumerate(timeline_subs):
        duration = time_seg["end"] - time_seg["start"]

        # Skip very short segments (less than 0.5s)
        # They'll be merged with adjacent segments
        if duration < 0.5:
            continue

        # Calculate how many characters this time segment should have
        target_chars = int(duration * chars_per_second)

        # Collect semantic segments for this time period
        segment_text = ""
        used_chars = 0

        # Add complete semantic segments until we reach target
        while (
            segment_idx < len(ref_segments)
            and used_chars + len(ref_segments[segment_idx]) <= target_chars
        ):
            segment_text += ref_segments[segment_idx] + " "
            used_chars += len(ref_segments[segment_idx]) + 1  # +1 for space
            segment_idx += 1

        # Clean up trailing space
        segment_text = segment_text.strip()

        # If segment is empty (shouldn't happen often), skip
        if not segment_text:
            continue

        # Check if segment exceeds max_chars
        if len(segment_text) > max_chars:
            # Split into multiple subtitles, try to preserve semantic boundaries
            words = segment_text.split(" ")
            current_chunk = ""
            chunks = []

            for word in words:
                if len(current_chunk) + len(word) + 1 <= max_chars:
                    current_chunk += (word + " ") if current_chunk else word
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = word

            if current_chunk:
                chunks.append(current_chunk.strip())

            # If chunks is empty (single word exceeds max_chars), force split
            if not chunks:
                chunks = [segment_text[:max_chars], segment_text[max_chars:]]

            # Assign each chunk a portion of the time segment
            chunk_duration = duration / len(chunks)
            for j, chunk in enumerate(chunks):
                chunk_start = time_seg["start"] + j * chunk_duration
                chunk_end = chunk_start + chunk_duration
                if chunk:  # Only add non-empty chunks
                    aligned_subs.append(
                        {"start": chunk_start, "end": chunk_end, "text": chunk}
                    )
        else:
            # Single subtitle for this time segment
            aligned_subs.append(
                {
                    "start": time_seg["start"],
                    "end": time_seg["end"],
                    "text": segment_text,
                }
            )

    # Step 3: Handle remaining reference text (if any)
    if segment_idx < len(ref_segments):
        # There's leftover text, append it to the last subtitle or create new ones
        remaining_text = " ".join(ref_segments[segment_idx:])

        # Check if the last subtitle can absorb remaining text
        if (
            aligned_subs
            and len(aligned_subs[-1]["text"]) + len(remaining_text) <= max_chars
        ):
            aligned_subs[-1]["text"] += " " + remaining_text
        else:
            # Create new subtitles for remaining text
            if aligned_subs:
                last_end = aligned_subs[-1]["end"]
            else:
                last_end = 0

            # Split into max_chars chunks
            chunks = []
            for j in range(0, len(remaining_text), max_chars):
                chunk = remaining_text[j : j + max_chars].strip()
                if chunk:
                    chunks.append(chunk)

            chunk_duration = 2.0  # Default duration
            for chunk in chunks:
                aligned_subs.append(
                    {"start": last_end, "end": last_end + chunk_duration, "text": chunk}
                )
                last_end += chunk_duration

    # Step 4: Merge very short consecutive subtitles
    # If a subtitle is very short (< 3 chars), merge it with previous or next
    i = 1
    while i < len(aligned_subs):
        current = aligned_subs[i]
        prev = aligned_subs[i - 1]

        # If current is too short, try to merge with previous
        if len(current["text"]) < 3 and i > 0:
            combined_text = prev["text"] + " " + current["text"]
            if len(combined_text) <= max_chars:
                # Merge into previous
                prev["text"] = combined_text
                prev["end"] = current["end"]
                aligned_subs.pop(i)  # Remove current
                continue  # Don't increment i

        i += 1

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
    """Test alignment function with provided data."""

    print("=" * 80)
    print("Testing Semantic-Aware Reference Text Alignment")
    print("=" * 80)
    print(f"\nReference text length: {len(REFERENCE_TEXT)} characters")
    print(f"Timeline segments: {len(MOCK_TIMELINE_SUBS)}")
    print(f"Total duration: {MOCK_TIMELINE_SUBS[-1]['end']} seconds")
    print()

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

    # Collect all text from aligned subs
    all_aligned_text = " ".join(sub["text"] for sub in aligned_subs)

    # Remove bracket tags from reference for comparison
    ref_clean = re.sub(r"\[[^\]]*\]", "", REFERENCE_TEXT)
    ref_clean = re.sub(r"\s+", " ", ref_clean).strip()

    print(f"\n1. Text Coverage:")
    print(f"   Reference text: {len(ref_clean)} chars")
    print(f"   Aligned text:  {len(all_aligned_text)} chars")
    print(f"   Coverage: {len(all_aligned_text) / len(ref_clean) * 100:.1f}%")

    # Check for repetition
    print(f"\n2. Repetition Check:")
    unique_subs = len(set(sub["text"] for sub in aligned_subs))
    print(f"   Total subs: {len(aligned_subs)}")
    print(f"   Unique subs: {unique_subs}")
    print(f"   Duplicates: {len(aligned_subs) - unique_subs}")
    if len(aligned_subs) - unique_subs > 5:
        print("   ❌ TOO MANY DUPLICATES!")
    else:
        print("   ✓ Acceptable duplicate count")

    # Check max_chars
    print(f"\n3. Max Chars Check:")
    exceeded = [sub for sub in aligned_subs if len(sub["text"]) > 20]
    print(f"   Subs exceeding 20 chars: {len(exceeded)}")
    if exceeded:
        print("   ❌ MAX CHARS VIOLATION!")
        for sub in exceeded[:5]:  # Show first 5
            print(f"     - [{sub['text']}] ({len(sub['text'])} chars)")
    else:
        print("   ✓ All subs within 20 chars")

    # Check for empty/short subs
    print(f"\n4. Quality Check:")
    short_subs = [sub for sub in aligned_subs if len(sub["text"]) < 3]
    print(f"   Subs with < 3 chars: {len(short_subs)}")
    if short_subs:
        print("   ❌ TOO MANY SHORT SUBS!")
        for sub in short_subs[:5]:
            print(f"     - [{sub['text']}]")
    else:
        print("   ✓ All subs have reasonable length")

    # Check for semantic coherence
    print(f"\n5. Semantic Coherence Check:")
    # Check if common problematic patterns exist
    cross_sentence_issues = []
    for i, sub in enumerate(aligned_subs):
        text = sub["text"]
        # Check for patterns that indicate cross-sentence cuts
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

    # Display full aligned text for manual inspection
    print(f"\n6. Full Aligned Text (first 400 chars):")
    print("-" * 80)
    print(all_aligned_text[:400])
    print("-" * 80)

    print("\n" + "=" * 80)
    print("Test Complete")
    print("=" * 80)

    # Generate SRT format output
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

    # Save SRT output
    with open("test_output_semantic.srt", "w", encoding="utf-8") as f:
        f.write(srt_output)
    print("\n✓ SRT output saved to: test_output_semantic.srt")

    return aligned_subs


if __name__ == "__main__":
    test_alignment()
