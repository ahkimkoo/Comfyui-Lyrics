#!/usr/bin/env python3
"""
Standalone test script for reference text alignment fix.
Tests the new align_text_to_timeline logic without ComfyUI dependencies.
"""

import re


# Simplified version of the alignment function for testing
def align_text_to_timeline(timeline_subs, reference_text, max_chars=20):
    """
    Align reference text to Whisper timeline using time-based distribution.
    """
    ref_text = reference_text.strip()
    if not ref_text:
        return timeline_subs

    # Remove bracket tags
    ref_text_no_tags = re.sub(r"\[[^\]]*\]", "", ref_text)

    # Keep spaces and newlines for now to preserve text flow
    ref_clean = ref_text_no_tags.replace("\n", " ").replace("\t", " ").strip()
    ref_clean = re.sub(r"\s+", " ", ref_clean)  # Normalize whitespace

    if not timeline_subs:
        return []

    # Calculate total audio duration
    total_duration = timeline_subs[-1]["end"] - timeline_subs[0]["start"]

    # Calculate average characters per second
    ref_len = len(ref_clean)
    chars_per_second = ref_len / total_duration if total_duration > 0 else 10

    aligned_subs = []
    text_position = 0  # Current position in reference text

    for sub in timeline_subs:
        duration = sub["end"] - sub["start"]
        if duration <= 0:
            continue

        # Calculate how many characters this segment should have
        target_chars = int(duration * chars_per_second)

        # Ensure we don't exceed max_chars per line
        target_chars = min(target_chars, max_chars)

        # Ensure we don't go beyond reference text
        if text_position >= ref_len:
            break

        # Extract substring for this segment
        end_position = min(text_position + target_chars, ref_len)
        segment_text = ref_clean[text_position:end_position]

        # Handle long segments that exceed max_chars
        if len(segment_text) > max_chars:
            # Split into multiple subtitles
            chunks = []
            for i in range(0, len(segment_text), max_chars):
                chunk = segment_text[i : i + max_chars].strip()
                if chunk:
                    chunks.append(chunk)

            if chunks:
                chunk_duration = duration / len(chunks)
                for i, chunk in enumerate(chunks):
                    chunk_start = sub["start"] + i * chunk_duration
                    chunk_end = chunk_start + chunk_duration
                    aligned_subs.append(
                        {"start": chunk_start, "end": chunk_end, "text": chunk}
                    )
                text_position = end_position
        else:
            # Single subtitle for this segment
            if segment_text.strip():
                aligned_subs.append(
                    {
                        "start": sub["start"],
                        "end": sub["end"],
                        "text": segment_text.strip(),
                    }
                )
            text_position = end_position

    # Handle any remaining text (if reference text is longer than audio timeline)
    if text_position < ref_len:
        remaining_text = ref_clean[text_position:]
        if aligned_subs:
            last_end = aligned_subs[-1]["end"]
        else:
            last_end = timeline_subs[-1]["end"] if timeline_subs else 0

        # Distribute remaining text in chunks of max_chars
        chunks = []
        for i in range(0, len(remaining_text), max_chars):
            chunk = remaining_text[i : i + max_chars].strip()
            if chunk:
                chunks.append(chunk)

        chunk_duration = 2.0  # Default duration for remaining chunks
        for i, chunk in enumerate(chunks):
            aligned_subs.append(
                {"start": last_end, "end": last_end + chunk_duration, "text": chunk}
            )
            last_end += chunk_duration

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
    """Test the alignment function with the provided data."""

    print("=" * 80)
    print("Testing Reference Text Alignment Fix")
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

    # Check for text coherence
    print(f"\n5. Text Coherence Check:")
    full_aligned_text = "".join(sub["text"] for sub in aligned_subs)
    print(f"   Full aligned text length: {len(full_aligned_text)} chars")
    print(
        f"   Starts with reference text? {full_aligned_text.startswith(ref_clean[:50])}"
    )

    # Display full aligned text for manual inspection
    print(f"\n6. Full Aligned Text (first 300 chars):")
    print("-" * 80)
    print(full_aligned_text[:300])
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
    with open("test_output.srt", "w", encoding="utf-8") as f:
        f.write(srt_output)
    print("\n✓ SRT output saved to: test_output.srt")

    return aligned_subs


if __name__ == "__main__":
    test_alignment()
