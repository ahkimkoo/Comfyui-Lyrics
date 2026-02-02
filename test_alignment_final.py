#!/usr/bin/env python3
"""
Test script for align_text_to_timeline function from nodes.py
This tests the actual implementation with user's reference text.
"""

import sys

sys.path.insert(0, "/var/tmp/vibe-kanban/worktrees/c874-bug/Comfyui-Lyrics")

from nodes import LyricsScrollEffect

# User's reference text (the complete correct text)
REFERENCE_TEXT = """[00:00.000] 开 专家说牛市马上就到来
[00:04.000] 的气概 心跳在加速 呼吸都停摆
[00:08.000] 梭哈是一种过人的气概
[00:12.000] 我要梭哈 所有的积蓄全买
[00:16.000] 停摆 我要梭哈 所有的积蓄全买
[00:20.000] 海 身边的兄弟都成了韭菜
[00:24.000] 菜 谁都逃不过被割的悲哀
[00:28.000] 哀 可惜我买在了最高处
[00:32.000] 处 现在只能看着账户发呆"""


# Extract clean text (remove timestamps and keep lyrics)
def extract_lyrics_from_srt(srt_text):
    """Extract lyrics from SRT format text."""
    import re

    lines = []
    for line in srt_text.strip().split("\n"):
        # Remove timestamp lines [00:00.000]
        if not re.match(r"^\[\d{2}:\d{2}\.\d{3}\]", line):
            lines.append(line.strip())
    return " ".join(lines)


# Simulate Whisper timeline output (what would come from transcribe_audio)
# Based on user's original output structure
def create_mock_timeline():
    """Create mock timeline segments simulating Whisper output."""
    # These are approximate based on the SRT timestamps
    # Whisper returns segments with start, end, and text
    timeline = [
        {"start": 0.0, "end": 4.0, "text": "开 专家说牛市马上就到来"},
        {"start": 4.0, "end": 8.0, "text": "的气概 心跳在加速 呼吸都停摆"},
        {"start": 8.0, "end": 12.0, "text": "梭哈是一种过人的气概"},
        {"start": 12.0, "end": 16.0, "text": "我要梭哈 所有的积蓄全买"},
        {"start": 16.0, "end": 20.0, "text": "停摆 我要梭哈 所有的积蓄全买"},
        {"start": 20.0, "end": 24.0, "text": "海 身边的兄弟都成了韭菜"},
        {"start": 24.0, "end": 28.0, "text": "菜 谁都逃不过被割的悲哀"},
        {"start": 28.0, "end": 32.0, "text": "哀 可惜我买在了最高处"},
        {"start": 32.0, "end": 36.0, "text": "处 现在只能看着账户发呆"},
    ]
    return timeline


def format_timestamp(seconds):
    """Format seconds to SRT timestamp format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def print_srt(subtitles):
    """Print subtitles in SRT format."""
    for i, sub in enumerate(subtitles, 1):
        start_time = format_timestamp(sub["start"])
        end_time = format_timestamp(sub["end"])
        text = sub["text"]
        print(f"{i}")
        print(f"{start_time} --> {end_time}")
        print(text)
        print()


def analyze_output(subtitles):
    """Analyze output for common issues."""
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)

    issues_found = []

    # Check for very short subtitles (single characters)
    short_subs = [i for i, sub in enumerate(subtitles) if len(sub["text"]) < 3]
    if short_subs:
        issues_found.append(f"Found {len(short_subs)} very short subtitles (<3 chars):")
        for idx in short_subs:
            issues_found.append(f"  - Line {idx + 1}: '{subtitles[idx]['text']}'")

    # Check for cross-sentence fragments
    # (This is harder to detect automatically, but we can look for certain patterns)
    cross_sentence = []
    for i in range(len(subtitles) - 1):
        current_end = subtitles[i]["text"][-1]
        next_start = subtitles[i + 1]["text"][0]
        # If current doesn't end with punctuation and next starts with new phrase
        if current_end not in "，。！？、；：" and next_start.isalpha():
            cross_sentence.append(
                f"  - Line {i + 1} -> {i + 2}: '{subtitles[i]['text']}' -> '{subtitles[i + 1]['text']}'"
            )

    if cross_sentence:
        issues_found.append("Potential cross-sentence fragments detected:")
        issues_found.extend(cross_sentence[:5])  # Show first 5

    # Statistics
    avg_length = (
        sum(len(s["text"]) for s in subtitles) / len(subtitles) if subtitles else 0
    )
    print(f"Total subtitles: {len(subtitles)}")
    print(f"Average length: {avg_length:.1f} characters")
    print(f"Min length: {min(len(s['text']) for s in subtitles) if subtitles else 0}")
    print(f"Max length: {max(len(s['text']) for s in subtitles) if subtitles else 0}")

    if issues_found:
        print("\n⚠️  ISSUES FOUND:")
        for issue in issues_found:
            print(issue)
    else:
        print("\n✅ NO OBVIOUS ISSUES DETECTED")


def main():
    print("=" * 60)
    print("TESTING align_text_to_timeline FUNCTION")
    print("=" * 60)

    # Create instance of LyricsScrollEffect
    node = LyricsScrollEffect()

    # Get clean reference text
    reference_text = extract_lyrics_from_srt(REFERENCE_TEXT)
    print(f"\nReference text (clean):")
    print(reference_text)
    print()

    # Create mock timeline
    timeline = create_mock_timeline()
    print(f"Timeline segments: {len(timeline)}")
    for i, seg in enumerate(timeline):
        print(
            f"  {i + 1}. [{seg['start']:.1f}s - {seg['end']:.1f}s]: {seg['text'][:50]}"
        )
    print()

    # Test with max_chars=20 (default)
    print("\n" + "=" * 60)
    print("TEST 1: max_chars=20")
    print("=" * 60)
    result_20 = node.align_text_to_timeline(timeline, reference_text, max_chars=20)
    print_srt(result_20)
    analyze_output(result_20)

    # Test with max_chars=15 (stricter)
    print("\n" + "=" * 60)
    print("TEST 2: max_chars=15")
    print("=" * 60)
    result_15 = node.align_text_to_timeline(timeline, reference_text, max_chars=15)
    print_srt(result_15)
    analyze_output(result_15)

    # Test with max_chars=30 (more lenient)
    print("\n" + "=" * 60)
    print("TEST 3: max_chars=30")
    print("=" * 60)
    result_30 = node.align_text_to_timeline(timeline, reference_text, max_chars=30)
    print_srt(result_30)
    analyze_output(result_30)


if __name__ == "__main__":
    main()
