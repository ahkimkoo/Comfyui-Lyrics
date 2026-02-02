#!/usr/bin/env python3
"""
Standalone test for align_text_to_timeline function
This extracts just the function for testing without ComfyUI dependencies.
"""


def align_text_to_timeline(timeline_subs, reference_text, max_chars=20):
    """
    Align reference text to Whisper timeline using semantic unit distribution (方案A - 完全修复).

    Strategy:
    1. Split reference text into complete semantic units by punctuation
    2. Distribute these units evenly across timeline segments
    3. Each time segment gets complete sentences/phrases
    4. If a unit is too long (> max_chars), split it by max_chars
    5. Merge very short (< 3 chars) subtitles with adjacent ones

    Args:
        timeline_subs: List of {'start', 'end', 'text'} from Whisper
        reference_text: The correct/original text to use
        max_chars: Maximum characters per subtitle line

    Returns:
        List of {'start', 'end', 'text'} with aligned reference text
    """
    import re

    ref_text = reference_text.strip()
    if not ref_text:
        return timeline_subs

    # Step 1: Remove bracket tags
    ref_text_no_tags = re.sub(r"\[[^\]]*\]", "", ref_text)

    # Step 2: Normalize whitespace to single spaces
    ref_clean = ref_text_no_tags.replace("\n", " ").replace("\t", " ")
    ref_clean = re.sub(r"\s+", " ", ref_clean).strip()

    if not timeline_subs:
        return []

    # Step 3: Split reference text into semantic units
    # Split by Chinese punctuation to preserve sentence boundaries
    # Pattern: ，。！？、；： keeps the delimiter with the preceding text
    semantic_chunks = re.split(r"([，。！？、；：])", ref_clean)

    # Re-attach punctuation to create complete semantic units
    # This ensures each unit ends with punctuation (complete thought)
    semantic_units = []
    i = 0
    while i < len(semantic_chunks):
        if i + 1 < len(semantic_chunks) and semantic_chunks[i + 1] in "，。！？、；：":
            # Combine text with its following punctuation
            combined = (semantic_chunks[i] + semantic_chunks[i + 1]).strip()
            if combined:
                semantic_units.append(combined)
            i += 2
        else:
            # Regular text segment (no following punctuation)
            if semantic_chunks[i].strip():
                semantic_units.append(semantic_chunks[i].strip())
            i += 1

    # Fallback: If no punctuation found, split into reasonable chunks
    # This handles cases where reference text has no punctuation marks
    if len(semantic_units) <= 1 and len(ref_clean) > max_chars:
        # No meaningful semantic units - split by character count
        # Try to split at word boundaries (spaces) where possible
        chunks = ref_clean.split(" ")
        semantic_units = []
        current_chunk = ""
        for chunk in chunks:
            if len(current_chunk + " " + chunk) <= max_chars:
                current_chunk = current_chunk + " " + chunk if current_chunk else chunk
            else:
                if current_chunk:
                    semantic_units.append(current_chunk.strip())
                current_chunk = chunk
        if current_chunk:
            semantic_units.append(current_chunk.strip())

    # Step 4: Distribute semantic units across timeline
    num_timeline_segments = len(timeline_subs)
    num_units = len(semantic_units)

    # Calculate how many units each time segment should get
    units_per_segment = []
    for seg_idx in range(num_timeline_segments):
        # Calculate proportional distribution
        start_ratio = seg_idx / num_timeline_segments
        end_ratio = (seg_idx + 1) / num_timeline_segments

        # Calculate unit range for this segment
        start_unit = int(seg_idx * num_units / num_timeline_segments)
        end_unit = int((seg_idx + 1) * num_units / num_timeline_segments)

        # Assign units to segments
        units_for_segment = []
        for unit_idx in range(start_unit, end_unit):
            if unit_idx < num_units:
                units_for_segment.append(semantic_units[unit_idx])

        units_per_segment.append(units_for_segment)

    aligned_subs = []

    # Step 5: Create subtitles from semantic units
    for seg_idx, time_seg in enumerate(timeline_subs):
        duration = time_seg["end"] - time_seg["start"]

        # Skip zero-duration segments
        if duration <= 0:
            continue

        units = units_per_segment[seg_idx]

        if not units:
            continue

        # Build segment text from units
        segment_text = ""
        for unit in units:
            segment_text += unit + " "

        # Clean up trailing space
        segment_text = segment_text.strip()

        # Skip if empty
        if not segment_text:
            continue

        # If segment text exceeds max_chars, split it
        if len(segment_text) > max_chars:
            # Split into max_chars-sized chunks
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

        else:
            # Single subtitle for this time segment
            aligned_subs.append(
                {
                    "start": time_seg["start"],
                    "end": time_seg["end"],
                    "text": segment_text,
                }
            )

    # Step 6: Handle remaining units (if any)
    # Calculate units assigned
    total_assigned = sum(len(units) for units in units_per_segment)

    if total_assigned < num_units:
        remaining_units = semantic_units[total_assigned:]

        if remaining_units:
            # Try to add to last subtitle
            if (
                aligned_subs
                and len(aligned_subs[-1]["text"]) + len(remaining_units[0]) + 1
                <= max_chars
            ):
                for unit in remaining_units:
                    if len(aligned_subs[-1]["text"]) + len(unit) + 1 <= max_chars:
                        aligned_subs[-1]["text"] += " " + unit
            else:
                # Create new subtitles for remaining units
                if aligned_subs:
                    last_end = aligned_subs[-1]["end"]
                else:
                    last_end = 0

                for unit in remaining_units:
                    aligned_subs.append(
                        {"start": last_end, "end": last_end + 2.0, "text": unit}
                    )
                    last_end += 2.0

    # Step 7: Post-processing: merge very short (< 3 chars) subtitles
    i = 1
    while i < len(aligned_subs):
        current = aligned_subs[i]
        prev = aligned_subs[i - 1]

        # Try merging very short subtitle with previous
        if len(current["text"]) < 3:
            combined = prev["text"] + " " + current["text"]
            if len(combined) <= max_chars:
                # Merge into previous
                prev["text"] = combined
                prev["end"] = current["end"]
                aligned_subs.pop(i)
                continue  # Don't increment i, check same index again

        i += 1

    return aligned_subs


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
    """Extract lyrics from SRT format text (handles inline timestamps)."""
    import re

    lines = []
    for line in srt_text.strip().split("\n"):
        # Remove inline timestamp like [00:00.000] from the beginning of the line
        clean_line = re.sub(r"^\[\d{2}:\d{2}\.\d{3}\]\s*", "", line)
        if clean_line.strip():
            lines.append(clean_line.strip())
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
    result_20 = align_text_to_timeline(timeline, reference_text, max_chars=20)
    print_srt(result_20)
    analyze_output(result_20)

    # Test with max_chars=15 (stricter)
    print("\n" + "=" * 60)
    print("TEST 2: max_chars=15")
    print("=" * 60)
    result_15 = align_text_to_timeline(timeline, reference_text, max_chars=15)
    print_srt(result_15)
    analyze_output(result_15)

    # Test with max_chars=30 (more lenient)
    print("\n" + "=" * 60)
    print("TEST 3: max_chars=30")
    print("=" * 60)
    result_30 = align_text_to_timeline(timeline, reference_text, max_chars=30)
    print_srt(result_30)
    analyze_output(result_30)


if __name__ == "__main__":
    main()
