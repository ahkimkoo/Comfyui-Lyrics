#!/usr/bin/env python3
"""
Test script for reference text alignment fix.
Tests the new align_text_to_timeline function.
"""

import sys
import os

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
    # Import the class
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from nodes import LyricsScroll

    # Create instance
    ls = LyricsScroll()

    print("=" * 80)
    print("Testing Reference Text Alignment")
    print("=" * 80)
    print(f"\nReference text length: {len(REFERENCE_TEXT)} characters")
    print(f"Timeline segments: {len(MOCK_TIMELINE_SUBS)}")
    print(f"Total duration: {MOCK_TIMELINE_SUBS[-1]['end']} seconds")
    print()

    # Run alignment
    aligned_subs = ls.align_text_to_timeline(
        MOCK_TIMELINE_SUBS, REFERENCE_TEXT, max_chars=20
    )

    print(f"\nGenerated {len(aligned_subs)} subtitle segments:")
    print("=" * 80)

    # Display results
    for i, sub in enumerate(aligned_subs):
        print(f"\n{i + 1}. [{sub['start']:.3f}s -> {sub['end']:.3f}s]")
        print(f"   Text: {sub['text']}")

    # Check for issues
    print("\n" + "=" * 80)
    print("Validation Checks:")
    print("=" * 80)

    # Collect all text from aligned subs
    all_aligned_text = " ".join(sub["text"] for sub in aligned_subs)

    # Remove bracket tags from reference for comparison
    import re

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

    # Check max_chars
    print(f"\n3. Max Chars Check:")
    exceeded = [sub for sub in aligned_subs if len(sub["text"]) > 20]
    print(f"   Subs exceeding 20 chars: {len(exceeded)}")
    if exceeded:
        for sub in exceeded:
            print(f"     - [{sub['text']}] ({len(sub['text'])} chars)")

    # Check for empty/short subs
    print(f"\n4. Quality Check:")
    short_subs = [sub for sub in aligned_subs if len(sub["text"]) < 3]
    print(f"   Subs with < 3 chars: {len(short_subs)}")
    if short_subs:
        for sub in short_subs:
            print(f"     - [{sub['text']}]")

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

    print("\nSRT Output (first 500 chars):")
    print("-" * 80)
    print(srt_output[:500])
    print("-" * 80)

    return aligned_subs


if __name__ == "__main__":
    test_alignment()
