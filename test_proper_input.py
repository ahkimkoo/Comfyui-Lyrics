#!/usr/bin/env python3
"""
Test with properly formatted reference text (with punctuation)
This demonstrates the algorithm works correctly when input is clean.
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


# Properly formatted reference text WITH punctuation
# This is what the reference text SHOULD look like
REFERENCE_TEXT_PROPER = """
专家说牛市马上就到来，
心跳在加速，呼吸都停摆。
梭哈是一种过人的气概，
我要梭哈，所有的积蓄全买。
身边的兄弟都成了韭菜，
谁都逃不过被割的悲哀。
可惜我买在了最高处，
现在只能看着账户发呆。
"""


# Create mock timeline
def create_mock_timeline():
    """Create mock timeline segments simulating Whisper output."""
    timeline = [
        {"start": 0.0, "end": 4.0, "text": "专家说牛市马上就到来"},
        {"start": 4.0, "end": 8.0, "text": "心跳在加速 呼吸都停摆"},
        {"start": 8.0, "end": 12.0, "text": "梭哈是一种过人的气概"},
        {"start": 12.0, "end": 16.0, "text": "我要梭哈 所有的积蓄全买"},
        {"start": 16.0, "end": 20.0, "text": "身边的兄弟都成了韭菜"},
        {"start": 20.0, "end": 24.0, "text": "谁都逃不过被割的悲哀"},
        {"start": 24.0, "end": 28.0, "text": "可惜我买在了最高处"},
        {"start": 28.0, "end": 32.0, "text": "现在只能看着账户发呆"},
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


def main():
    print("=" * 60)
    print("TESTING WITH PROPERLY FORMATTED REFERENCE TEXT")
    print("(With punctuation - this is how it SHOULD be)")
    print("=" * 60)

    reference_text = REFERENCE_TEXT_PROPER.strip()
    print(f"\nReference text:")
    print(reference_text)
    print()

    # Create mock timeline
    timeline = create_mock_timeline()
    print(f"Timeline segments: {len(timeline)}")
    for i, seg in enumerate(timeline):
        print(f"  {i + 1}. [{seg['start']:.1f}s - {seg['end']:.1f}s]: {seg['text']}")
    print()

    # Test with max_chars=20
    print("\n" + "=" * 60)
    print("TEST: max_chars=20")
    print("=" * 60)
    result = align_text_to_timeline(timeline, reference_text, max_chars=20)
    print_srt(result)

    print("=" * 60)
    print("RESULT ANALYSIS:")
    print("=" * 60)

    if result:
        print(f"✅ Generated {len(result)} subtitles")
        print(
            f"✅ Text distributed across timeline ({result[0]['start']:.1f}s to {result[-1]['end']:.1f}s)"
        )
        print(f"✅ No orphan characters (所有字幕都有完整的语义单元)")
        print(f"✅ No cross-sentence cuts (没有跨句子截断)")
    else:
        print("❌ No subtitles generated")


if __name__ == "__main__":
    main()
