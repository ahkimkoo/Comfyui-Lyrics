import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor
import folder_paths
import os
import re
import math
import gc
import tempfile
import random
import subprocess
import shutil
from datetime import datetime

# Attempt to import whisper
try:
    import whisper

    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

# Attempt to import torchaudio for resampling
try:
    import torchaudio

    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False


class LyricsScroll:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.font_cache = {}
        self.whisper_model = None
        self.whisper_model_name = None

    @classmethod
    def INPUT_TYPES(s):
        # Ensure fonts directory exists and list fonts
        font_dir = os.path.join(folder_paths.models_dir, "fonts")
        os.makedirs(font_dir, exist_ok=True)
        fonts = [
            f for f in os.listdir(font_dir) if f.lower().endswith((".ttf", ".otf"))
        ]
        if not fonts:
            fonts = ["Arial.ttf"]  # Fallback

        return {
            "required": {
                "width": ("INT", {"default": 720, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 1280, "min": 64, "max": 4096}),
                "margin_left": ("INT", {"default": 50, "min": 0, "max": 4096}),
                "margin_right": ("INT", {"default": 50, "min": 0, "max": 4096}),
                "y_pos": ("INT", {"default": 640, "min": 0, "max": 4096}),
                "font_size": (
                    "INT",
                    {"default": 30, "min": 10, "max": 200},
                ),  # Inactive size
                "active_font_size": (
                    "INT",
                    {"default": 40, "min": 10, "max": 200},
                ),  # Active size
                "letter_spacing": ("INT", {"default": 0, "min": -10, "max": 200}),
                "line_gap": (
                    "INT",
                    {"default": 20, "min": 0, "max": 1000},
                ),  # Gap between items
                "text_color": ("STRING", {"default": "#FFFFFF"}),
                "stroke_width": ("INT", {"default": 1, "min": 0, "max": 20}),
                "stroke_color": ("STRING", {"default": "#000000"}),
                "shadow_color": ("STRING", {"default": "#000000"}),
                "shadow_offset_x": ("INT", {"default": 2, "min": -100, "max": 100}),
                "shadow_offset_y": ("INT", {"default": 2, "min": -100, "max": 100}),
                "shadow_alpha": (
                    "FLOAT",
                    {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.1},
                ),
                "font_filename": (fonts,),
                "frame_rate": (
                    "FLOAT",
                    {"default": 25.0, "min": 1.0, "max": 120.0, "step": 0.01},
                ),
                "max_chars_per_line": ("INT", {"default": 20, "min": 5, "max": 100}),
                "batch_size": ("INT", {"default": 250, "min": 10, "max": 2000}),
                "whisper_prompt": (
                    ["简体中文", "繁体中文", "English", "日本語", "한국어"],
                ),
            },
            "optional": {
                "audio": ("AUDIO",),
                "lyrics": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "1\n00:00:00,000 --> 00:00:05,000\n第一行字幕\n\n2\n00:00:05,000 --> 00:00:10,000\n第二行字幕",
                        "forceInput": False,
                    },
                ),
                "reference_text": (
                    "STRING",
                    {"multiline": True, "default": "", "forceInput": False},
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "subtitles")
    FUNCTION = "generate_lyrics"
    CATEGORY = "Lyrics"

    def get_font(self, font_filename, size):
        key = (font_filename, size)
        if key not in self.font_cache:
            font_path = os.path.join(folder_paths.models_dir, "fonts", font_filename)
            try:
                self.font_cache[key] = ImageFont.truetype(font_path, size)
            except:
                self.font_cache[key] = ImageFont.load_default()
        return self.font_cache[key]

    def wrap_text(self, text, font, max_width, letter_spacing, line_gap):
        if not text:
            return [], 0, 0

        lines = []
        current_line = ""
        current_width = 0
        max_line_width = 0

        # Simple char-by-char wrap for Chinese/English mixed
        for char in text:
            # Measure char width
            if hasattr(font, "getlength"):
                char_w = font.getlength(char)
            else:
                try:
                    char_w, _ = font.getsize(char)
                except:
                    char_w = font.size * 0.5

            char_w += letter_spacing

            if current_width + char_w > max_width and current_line:
                lines.append(current_line)
                max_line_width = max(max_line_width, current_width)
                current_line = char
                current_width = char_w
            else:
                current_line += char
                current_width += char_w

        if current_line:
            lines.append(current_line)
            max_line_width = max(max_line_width, current_width)

        # Height estimation
        try:
            ascent, descent = font.getmetrics()
            line_height = ascent + descent
        except:
            line_height = font.size * 1.2

        # Calculate total height including line_gap for multiline
        if len(lines) > 0:
            total_height = line_height * len(lines) + line_gap * (len(lines) - 1)
        else:
            total_height = 0

        return lines, max_line_width, total_height

    def draw_multiline_text(
        self,
        draw,
        x,
        y,
        lines,
        font,
        fill,
        stroke_width,
        stroke_fill,
        shadow_fill,
        shadow_offset,
        letter_spacing,
        line_gap,
    ):
        # Calculate line height
        try:
            ascent, descent = font.getmetrics()
            base_h = ascent + descent
        except:
            base_h = font.size * 1.2

        # Use line_gap for spacing between lines
        line_spacing = base_h + line_gap

        current_y = y
        for line in lines:
            self.draw_line_custom(
                draw,
                x,
                current_y,
                line,
                font,
                fill,
                stroke_width,
                stroke_fill,
                shadow_fill,
                shadow_offset,
                letter_spacing,
            )
            current_y += line_spacing

    def draw_line_custom(
        self,
        draw,
        x,
        y,
        text,
        font,
        fill,
        stroke_width,
        stroke_fill,
        shadow_fill,
        shadow_offset,
        letter_spacing,
    ):
        # Draw Shadow if enabled (alpha > 0)
        # shadow_fill is RGBA tuple.
        if (
            shadow_fill
            and shadow_fill[3] > 0
            and (shadow_offset[0] != 0 or shadow_offset[1] != 0)
        ):
            sx, sy = shadow_offset
            if letter_spacing == 0:
                draw.text(
                    (x + sx, y + sy), text, font=font, fill=shadow_fill, stroke_width=0
                )
            else:
                cursor_x = x
                for char in text:
                    if hasattr(font, "getlength"):
                        w = font.getlength(char)
                    else:
                        w = font.size * 0.5  # fallback

                    draw.text(
                        (cursor_x + sx, y + sy),
                        char,
                        font=font,
                        fill=shadow_fill,
                        stroke_width=0,
                    )
                    cursor_x += w + letter_spacing

        # Draw Main Text
        if letter_spacing == 0:
            if stroke_width > 0:
                draw.text(
                    (x, y),
                    text,
                    font=font,
                    fill=stroke_fill,
                    stroke_width=stroke_width,
                    stroke_fill=stroke_fill,
                )
            draw.text((x, y), text, font=font, fill=fill, stroke_width=0)
            return

        cursor_x = x
        for char in text:
            if hasattr(font, "getlength"):
                w = font.getlength(char)
            else:
                try:
                    w, h = font.getsize(char)
                except:
                    w = font.size * 0.5

            if stroke_width > 0:
                draw.text(
                    (cursor_x, y),
                    char,
                    font=font,
                    fill=stroke_fill,
                    stroke_width=stroke_width,
                    stroke_fill=stroke_fill,
                )
            draw.text((cursor_x, y), char, font=font, fill=fill, stroke_width=0)

            cursor_x += w + letter_spacing

    def parse_srt(self, srt_text):
        pattern = re.compile(
            r"(\d+)\n(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})\n(.*?)(?=\n\n|\n$|$)",
            re.DOTALL,
        )
        matches = pattern.findall(srt_text.replace("\r\n", "\n") + "\n\n")

        subs = []
        for match in matches:
            idx, h1, m1, s1, ms1, h2, m2, s2, ms2, content = match
            start = int(h1) * 3600 + int(m1) * 60 + int(s1) + int(ms1) / 1000
            end = int(h2) * 3600 + int(m2) * 60 + int(s2) + int(ms2) / 1000
            content = content.strip()
            subs.append({"start": start, "end": end, "text": content})
        return subs

    def align_text_to_timeline(self, timeline_subs, reference_text, max_chars=20):
        """
        Align reference text to Whisper timeline using fuzzy matching.

        Uses Whisper's detected timing segments and matches them to the
        reference text using sliding window similarity matching. This handles
        cases where Whisper misrecognizes words (extra/missing/wrong characters).

        Ignores:
        - Whitespace (spaces, newlines, tabs)
        - Bracket tags like [Chorus], [Verse], [Intro bass, guitar], etc.

        Args:
            timeline_subs: List of {'start', 'end', 'text'} from Whisper
            reference_text: The correct/original text to use
            max_chars: Maximum characters per subtitle line

        Returns:
            List of {'start', 'end', 'text'} with aligned reference text
        """
        from difflib import SequenceMatcher
        import re

        ref_text = reference_text.strip()
        if not ref_text:
            return timeline_subs

        # Remove bracket tags (e.g., [Chorus], [Verse 1], [Intro bass, guitar])
        # Pattern matches [...] and removes it
        ref_text_no_tags = re.sub(r"\[[^\]]*\]", "", ref_text)

        # Remove all whitespace from reference for better matching
        ref_clean = (
            ref_text_no_tags.replace(" ", "")
            .replace("\n", "")
            .replace("\t", "")
            .replace("\r", "")
        )
        ref_len = len(ref_clean)

        # Build position mapping: clean_index -> original_index in ref_text_no_tags
        clean_to_orig_no_tags = []
        orig_idx = 0
        for char in ref_text_no_tags:
            if not char.isspace():
                clean_to_orig_no_tags.append(orig_idx)
            orig_idx += 1

        # Pre-extract timing from Whisper segments
        segments = []
        for sub in timeline_subs:
            clean_text = sub["text"].replace(" ", "").replace("\n", "")
            if clean_text:
                segments.append(
                    {
                        "start": sub["start"],
                        "end": sub["end"],
                        "duration": sub["end"] - sub["start"],
                        "text": clean_text,
                    }
                )

        if not segments:
            # Fallback to simple splitting
            return self._split_reference_by_time(
                ref_text_no_tags, timeline_subs, max_chars
            )

        total_audio_duration = segments[-1]["end"] - segments[0]["start"]

        aligned_subs = []
        ref_pos = 0  # Position in clean text

        for seg_idx, seg in enumerate(segments):
            if ref_pos >= ref_len:
                break

            seg_len = len(seg["text"])
            if seg_len == 0:
                continue

            # Calculate expected position based on time
            time_ratio = (
                seg["duration"] / total_audio_duration
                if total_audio_duration > 0
                else 0
            )
            expected_pos_by_time = (
                int(
                    ref_len
                    * (seg["start"] - segments[0]["start"])
                    / total_audio_duration
                )
                if total_audio_duration > 0
                else ref_pos
            )

            # Use both position heuristics
            expected_pos = max(ref_pos, expected_pos_by_time - max_chars)
            expected_pos = min(expected_pos, ref_len - max_chars)

            # Determine window size
            window_size = min(max_chars * 2, max(seg_len, max_chars))

            best_match = None
            best_score = 0
            best_end = ref_pos
            best_start = ref_pos

            search_start = max(0, expected_pos - window_size)
            search_end = min(ref_len, expected_pos + window_size * 2)

            for start in range(search_start, search_end):
                for size_adjust in [-2, -1, 0, 1, 2]:
                    window_len = max(5, window_size + size_adjust)
                    end = min(start + window_len, ref_len)
                    window = ref_clean[start:end]

                    if not window or len(window) < 3:
                        continue

                    similarity = SequenceMatcher(None, seg["text"], window).ratio()
                    pos_distance = abs(start - expected_pos)
                    pos_penalty = (pos_distance / ref_len) * 0.3
                    len_ratio = min(len(window), seg_len) / max(len(window), seg_len)
                    len_bonus = len_ratio * 0.1
                    adjusted_score = similarity - pos_penalty + len_bonus

                    if adjusted_score > best_score:
                        best_score = adjusted_score
                        best_match = window
                        best_end = end
                        best_start = start

                        if similarity > 0.9:
                            break
                else:
                    continue
                break

            if best_match:
                # Use position mapping to get original text (without bracket tags)
                orig_start = (
                    clean_to_orig_no_tags[best_start]
                    if best_start < len(clean_to_orig_no_tags)
                    else 0
                )

                # Extract from original (without bracket tags), but skip leading whitespace
                aligned_text = ref_text_no_tags[orig_start:]
                # Strip leading whitespace and take up to max_chars
                aligned_text = aligned_text.lstrip()[:max_chars]

                # Update ref_pos
                ref_pos = best_end

                if aligned_text.strip():
                    aligned_subs.append(
                        {
                            "start": seg["start"],
                            "end": seg["end"],
                            "text": aligned_text.strip(),
                        }
                    )

        # Handle any remaining reference text
        if ref_pos < ref_len and ref_pos < len(clean_to_orig_no_tags):
            orig_start = (
                clean_to_orig_no_tags[ref_pos]
                if ref_pos < len(clean_to_orig_no_tags)
                else 0
            )
            remaining = ref_text_no_tags[orig_start:]
            if aligned_subs:
                last_end = aligned_subs[-1]["end"]
            else:
                last_end = 0

            for i in range(0, len(remaining), max_chars):
                chunk = remaining[i : i + max_chars].strip()
                if chunk:
                    aligned_subs.append(
                        {"start": last_end, "end": last_end + 2.0, "text": chunk}
                    )
                    last_end += 2.0

        return aligned_subs

    def _split_reference_by_time(self, ref_text, timeline_subs, max_chars):
        """Fallback: simply split reference text by time segments."""
        if not timeline_subs:
            return []

        ref_len = len(ref_text)
        total_duration = timeline_subs[-1]["end"] - timeline_subs[0]["start"]
        chars_per_second = (
            ref_len / total_duration if total_duration > 0 else max_chars / 2
        )

        aligned_subs = []
        text_index = 0

        for sub in timeline_subs:
            duration = sub["end"] - sub["start"]
            target_chars = min(max_chars, max(1, int(duration * chars_per_second)))

            if text_index < ref_len:
                chunk = ref_text[text_index : text_index + target_chars]
                text_index += len(chunk)
                if chunk.strip():
                    aligned_subs.append(
                        {
                            "start": sub["start"],
                            "end": sub["end"],
                            "text": chunk.strip(),
                        }
                    )

        return aligned_subs

    def _map_clean_to_original(self, original, cleaned, clean_pos, length):
        """Map a position in cleaned text back to original text."""
        clean_idx = 0
        orig_idx = 0
        target_clean_pos = clean_pos
        target_length = 0

        while orig_idx < len(original) and clean_idx < cleaned:
            if original[orig_idx].isspace():
                orig_idx += 1
                continue

            if clean_idx >= target_clean_pos:
                return orig_idx

            clean_idx += 1
            orig_idx += 1

        return max(0, target_clean_pos)

    def _update_ref_pos(self, cleaned, current_pos, advance):
        """Update reference position accounting for removed whitespace."""
        # Count non-space characters to advance
        count = 0
        pos = current_pos
        while pos < len(cleaned) and count < advance:
            pos += 1
            count += 1
        return pos

    def transcribe_audio(self, audio_data, max_chars=20, prompt="简体中文"):
        if not WHISPER_AVAILABLE:
            raise ImportError(
                "OpenAI Whisper is not installed. Please install it to generate subtitles automatically."
            )

        waveform = audio_data["waveform"]
        sample_rate = audio_data["sample_rate"]

        print(
            f"LyricsScroll: transcribe_audio input - waveform.shape={waveform.shape}, sample_rate={sample_rate}"
        )

        # Handle different audio formats
        # ComfyUI audio format can be:
        # - 3D: (batch, channels, samples) -> remove batch -> (channels, samples)
        # - 2D: (channels, samples) or (samples, channels) -> convert to mono
        # - 1D: (samples) -> already mono

        if waveform.dim() == 3:
            # (batch, channels, samples) -> (channels, samples)
            original_waveform = waveform
            waveform = waveform[0]
            del original_waveform
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"LyricsScroll: Removed batch dimension, shape={waveform.shape}")

        if waveform.dim() == 2:
            # Determine if format is (channels, samples) or (samples, channels)
            # Usually in audio: first dim is channels (small number like 1 or 2)
            # second dim is samples (large number like millions)
            original_waveform = waveform
            if waveform.shape[0] <= waveform.shape[1]:
                # Format is (channels, samples) - e.g., (2, 7066584)
                # Average across channels to get mono: (samples,)
                waveform = torch.mean(waveform, dim=0)
                print(
                    f"LyricsScroll: Converted (channels, samples) to mono, shape={waveform.shape}"
                )
            else:
                # Format is (samples, channels) - e.g., (7066584, 2)
                # Average across channels to get mono: (samples,)
                waveform = torch.mean(waveform, dim=1)
                print(
                    f"LyricsScroll: Converted (samples, channels) to mono, shape={waveform.shape}"
                )
            del original_waveform
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"LyricsScroll: After preprocessing - waveform.shape={waveform.shape}")

        if TORCHAUDIO_AVAILABLE and sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            del resampler
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        elif sample_rate != 16000:
            raise ImportError("Torchaudio required for resampling.")

        print("LyricsScroll: Loading Whisper model...")
        model_name = "base"
        if self.whisper_model is None or self.whisper_model_name != model_name:
            self.whisper_model = whisper.load_model(model_name, device=self.device)
            self.whisper_model_name = model_name
        model = self.whisper_model
        print("LyricsScroll: Transcribing...")

        # Infer language logic or keep zh? User only asked for prompt.
        # But if prompt is English, language='zh' is bad.
        lang = "zh"
        if "English" in prompt:
            lang = "en"
        elif "日本" in prompt:
            lang = "ja"
        elif "한국" in prompt:
            lang = "ko"

        result = model.transcribe(
            waveform.numpy(), language=lang, initial_prompt=prompt
        )

        subs = []
        for segment in result["segments"]:
            text = segment["text"].strip()
            start = segment["start"]
            end = segment["end"]

            if len(text) > max_chars:
                chunks = []
                curr = ""
                for char in text:
                    curr += char
                    if len(curr) >= max_chars:
                        chunks.append(curr)
                        curr = ""
                if curr:
                    chunks.append(curr)

                duration = end - start
                chunk_duration = duration / len(chunks)

                for i, chunk in enumerate(chunks):
                    s = start + i * chunk_duration
                    e = start + (i + 1) * chunk_duration
                    subs.append({"start": s, "end": e, "text": chunk})
            else:
                subs.append({"start": start, "end": end, "text": text})

        srt_output = ""
        for i, sub in enumerate(subs):
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

        return subs, srt_output

    def generate_lyrics(
        self,
        width,
        height,
        margin_left,
        margin_right,
        y_pos,
        font_size,
        active_font_size,
        letter_spacing,
        line_gap,
        text_color,
        stroke_width,
        stroke_color,
        shadow_color,
        shadow_offset_x,
        shadow_offset_y,
        shadow_alpha,
        font_filename,
        frame_rate,
        max_chars_per_line,
        batch_size,
        whisper_prompt,
        audio=None,
        lyrics="",
        reference_text="",
    ):
        self.font_cache = {}

        subs = []
        srt_text = ""

        # 1. Try to use provided lyrics (SRT format)
        if lyrics and lyrics.strip():
            print("LyricsScroll: Using provided lyrics/SRT.")
            subs = self.parse_srt(lyrics)
            srt_text = lyrics
            if not subs:
                print("LyricsScroll: Warning - Provided lyrics is not valid SRT.")

        # 2. If no SRT text, try Whisper (requires audio)
        use_reference_align = False
        if not subs:
            if audio is not None:
                if reference_text and reference_text.strip():
                    print(f"LyricsScroll: Using Whisper with reference text alignment.")
                    use_reference_align = True
                else:
                    print(
                        f"LyricsScroll: No lyrics provided. Running Whisper with max_chars={max_chars_per_line}, prompt='{whisper_prompt}'..."
                    )
                subs, srt_text = self.transcribe_audio(
                    audio, max_chars=max_chars_per_line, prompt=whisper_prompt
                )
            else:
                raise ValueError(
                    "LyricsScroll: Either 'audio' or 'lyrics' (SRT) must be provided. If providing lyrics only, ensure it is valid SRT format."
                )

        # 3. Apply reference text alignment if provided
        if use_reference_align and subs and reference_text and reference_text.strip():
            print(
                f"LyricsScroll: Aligning reference text ({len(reference_text)} chars) to timeline ({len(subs)} segments)..."
            )
            subs = self.align_text_to_timeline(
                subs, reference_text, max_chars=max_chars_per_line
            )
            # Regenerate SRT with aligned text
            srt_output = ""
            for i, sub in enumerate(subs):
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
            srt_text = srt_output
            print(f"LyricsScroll: Aligned to {len(subs)} subtitle segments.")

        # 4. Determine Duration
        if audio is not None:
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            print(
                f"LyricsScroll: Audio info - waveform.shape={waveform.shape}, sample_rate={sample_rate}"
            )
            total_samples = waveform.shape[-1]
            duration = total_samples / sample_rate
            # If audio waveform is empty but we have subs, use subs duration
            if total_samples == 0 and subs:
                print(
                    f"LyricsScroll: Warning - Audio waveform is empty! Using subtitle duration instead."
                )
                duration = subs[-1]["end"] + 2.0
        elif subs:
            # Fallback duration from subtitles if no audio
            duration = subs[-1]["end"] + 2.0  # Add padding
        else:
            duration = 10.0  # Should not happen

        total_frames = int(duration * frame_rate)

        try:
            text_color_rgb = ImageColor.getrgb(text_color)
            stroke_color_rgb = ImageColor.getrgb(stroke_color)
            shadow_color_rgb = ImageColor.getrgb(shadow_color)
        except:
            text_color_rgb = (255, 255, 255)
            stroke_color_rgb = (0, 0, 0)
            shadow_color_rgb = (0, 0, 0)

        output_images = []

        print(
            f"LyricsScroll: Generating {total_frames} frames with slot-based animation (batch_size={batch_size})..."
        )

        if not subs:
            empty = torch.zeros(
                (max(1, total_frames), height, width, 4), dtype=torch.float32
            )
            return (empty, srt_text)

        ANIMATION_DURATION = 0.3  # 300ms
        GAP = line_gap
        MAX_TEXT_WIDTH = max(100, width - margin_left - margin_right)

        # Pre-calculate layouts logic helper
        def calculate_layout(item_idx, is_target_state):
            # is_target_state=True: Item is in its "Active" destination configuration (Slot 2 for target, etc)
            # Layout logic:
            # We assume "Target State N" means: Item N is Active (Slot 2).
            # Item N-1 is Top (Slot 1).
            # Item N+1 is Bottom (Slot 3).

            # We need to know the properties (Size, Font) of items in this state.

            # Return dict: {idx: {y, scale, alpha}}

            layout = {}

            # Helper to get height
            def get_h(idx, size):
                if idx < 0 or idx >= len(subs):
                    # Dummy height for virtual items (-1, etc) to maintain spacing
                    # Use a standard single line height of the given size
                    font = self.get_font(font_filename, size)
                    try:
                        ascent, descent = font.getmetrics()
                        return ascent + descent
                    except:
                        return size * 1.2

                font = self.get_font(font_filename, size)
                _, _, h = self.wrap_text(
                    subs[idx]["text"], font, MAX_TEXT_WIDTH, letter_spacing, line_gap
                )
                return h

            # Central Item (Slot 2)
            h_center = get_h(item_idx, active_font_size)
            y_center = y_pos - h_center / 2

            # If item_idx is valid, it's visible opacity 255. If virtual (-1), 0.
            alpha_center = 255 if 0 <= item_idx < len(subs) else 0

            layout[item_idx] = {
                "y": y_center,
                "size": active_font_size,
                "alpha": alpha_center,
            }

            # Top Item (Slot 1) - Item N-1
            idx_top = item_idx - 1
            if idx_top >= 0:
                h_top = get_h(idx_top, font_size)
                # Position: Above center item.
                # y_top = y_center - GAP - h_top
                y_top = y_center - GAP - h_top
                layout[idx_top] = {"y": y_top, "size": font_size, "alpha": 150}

            # Bottom Item (Slot 3) - Item N+1
            idx_bot = item_idx + 1
            if idx_bot < len(subs):
                h_bot = get_h(idx_bot, font_size)
                # Position: Below center item
                y_bot = y_center + h_center + GAP
                layout[idx_bot] = {"y": y_bot, "size": font_size, "alpha": 150}

            # "Out" positions for transitions
            # Item N-2 (Top Out)
            idx_out_top = item_idx - 2
            if idx_out_top >= 0:
                h_out = get_h(idx_out_top, font_size)
                # Even further up
                # Need reference to y_top
                if idx_top >= 0:
                    y_out = layout[idx_top]["y"] - GAP - h_out
                else:
                    y_out = y_center - GAP - h_out  # Fallback
                layout[idx_out_top] = {"y": y_out, "size": font_size, "alpha": 0}

            # Item N+2 (Bottom In)
            idx_in_bot = item_idx + 2
            if idx_in_bot < len(subs):
                # Start further down
                if idx_bot < len(subs):
                    y_in = layout[idx_bot]["y"] + get_h(idx_bot, font_size) + GAP
                else:
                    y_in = y_center + h_center + GAP
                layout[idx_in_bot] = {"y": y_in, "size": font_size, "alpha": 0}

            return layout

        def generate_single_frame(frame_idx):
            """Generate a single frame at the given index"""
            f = frame_idx
            current_time = f / frame_rate

            # 1. Identify "Next" line to determine phase
            # next_idx is the first subtitle that hasn't started yet (start > current_time)
            next_idx = len(subs)
            for i, sub in enumerate(subs):
                if sub["start"] > current_time:
                    next_idx = i
                    break

            # current settled state is next_idx - 1
            # e.g. if next is 0 (first one hasn't started), state is -1 (pre-start)
            # if next is 1 (0 has started, 1 hasn't), state is 0.

            current_state_idx = next_idx - 1

            # 2. Check for transition to next_idx
            # Transition happens in [next_start - 0.3, next_start]

            layout_current = {}

            # Ease Out Cubic
            def ease_out_cubic(t):
                return 1 - math.pow(1 - t, 3)

            is_transitioning = False

            if next_idx < len(subs):
                next_start = subs[next_idx]["start"]
                anim_start = next_start - ANIMATION_DURATION

                if current_time >= anim_start:
                    # Transitioning from current_state_idx -> next_idx
                    is_transitioning = True
                    raw_t = (current_time - anim_start) / ANIMATION_DURATION
                    raw_t = max(0.0, min(1.0, raw_t))
                    t = ease_out_cubic(raw_t)

                    # Interpolate Layout(current_state_idx) -> Layout(next_idx)
                    layout_prev = calculate_layout(current_state_idx, True)
                    layout_next = calculate_layout(next_idx, True)

                    # Target index for triangle logic is the incoming one
                    target_idx = next_idx

                    all_keys = set(layout_prev.keys()) | set(layout_next.keys())
                    for k in all_keys:
                        # Defaults for entering/leaving items
                        # If k not in prev (Entering): It comes from Bottom (Slot 3 + Gap) or generic down
                        # If k not in next (Leaving): It goes to Top (Slot 1 - Gap) or generic up

                        # Better default calculation?
                        # Use y_pos +/- height as safe far-field
                        p1 = layout_prev.get(
                            k, {"y": y_pos + height / 2, "size": font_size, "alpha": 0}
                        )
                        p2 = layout_next.get(
                            k, {"y": y_pos - height / 2, "size": font_size, "alpha": 0}
                        )

                        curr_y = p1["y"] + (p2["y"] - p1["y"]) * t
                        curr_size = p1["size"] + (p2["size"] - p1["size"]) * t
                        curr_alpha = p1["alpha"] + (p2["alpha"] - p1["alpha"]) * t

                        layout_current[k] = {
                            "y": curr_y,
                            "size": curr_size,
                            "alpha": curr_alpha,
                        }

            if not is_transitioning:
                # Settled at current_state_idx
                layout_current = calculate_layout(current_state_idx, True)
                target_idx = current_state_idx
                t = 1.0

            # Render
            img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)

            # Sort by index to draw in order
            sorted_indices = sorted(layout_current.keys())

            for i in sorted_indices:
                props = layout_current[i]
                if props["alpha"] < 1:
                    continue

                this_font = self.get_font(font_filename, int(props["size"]))

                # Colors
                alpha_int = int(props["alpha"])
                this_text_color = text_color_rgb + (alpha_int,)
                this_stroke_color = stroke_color_rgb + (alpha_int,)

                # Shadow Color with combined alpha
                # Shadow base alpha * Text Fade alpha
                this_shadow_alpha = int(255 * shadow_alpha * (alpha_int / 255.0))
                this_shadow_color = shadow_color_rgb + (this_shadow_alpha,)

                # Text Wrapping
                lines, _, _ = self.wrap_text(
                    subs[i]["text"], this_font, MAX_TEXT_WIDTH, letter_spacing, line_gap
                )

                # Draw Triangle (Only for Active Target)
                # User: "到了二行后自动在行首插入三角符号"
                # "到了二行" means t=1.0 or settled state for target_idx.
                # During transition? "在移动的过程中下一行字幕会出现在三行".
                # Triangle should probably appear when it is THE active line.
                # If we are transitioning TO N, N has triangle?
                # Let's fade in triangle with t?
                # Or only show if i == target_idx?

                # Triangle Logic:
                # Always show on the current "Active Intent" line.
                # If transitioning N-1 -> N. N is the goal.
                # If t < 0.5, maybe focus is still N-1 visually?
                # But requirement says "Moving to second line... insert triangle".
                # Implies triangle appears when it settles?
                # Let's fade it in based on 't' if i == target_idx.
                # If i == target_idx - 1 (Old focus), fade out triangle.

                tri_alpha = 0
                if i == target_idx:
                    # Incoming focus
                    # If settled, 255. If transitioning, 0->255.
                    if t < 1.0:  # Transitioning
                        tri_alpha = int(255 * t)
                    else:
                        tri_alpha = 255
                elif target_idx > 0 and i == target_idx - 1:
                    # Outgoing focus
                    # If transitioning, 255 -> 0
                    if t < 1.0:
                        tri_alpha = int(255 * (1 - t))

                # Draw Triangle
                if tri_alpha > 10:
                    t_size = max(10, int(props["size"] * 0.5))
                    tri_x = margin_left
                    # Center Y relative to first line of text?
                    # Text Y is top-left.
                    # Calculate single line height for centering triangle on first line
                    ascent, descent = this_font.getmetrics()
                    first_line_h = ascent + descent
                    tri_y = props["y"] + first_line_h / 2

                    p1 = (tri_x + t_size, tri_y)
                    p2 = (tri_x, tri_y - t_size * 0.6)
                    p3 = (tri_x, tri_y + t_size * 0.6)

                    tri_fill = text_color_rgb + (tri_alpha,)
                    tri_stroke = stroke_color_rgb + (tri_alpha,)

                    # Fix: Only pass outline if stroke_width > 0
                    if stroke_width > 0:
                        draw.polygon([p1, p2, p3], fill=tri_fill, outline=tri_stroke)
                    else:
                        draw.polygon([p1, p2, p3], fill=tri_fill, outline=None)

                # Draw Text
                # Indent if triangle is present?
                # User said: "Triangle inserted at head".
                # Previous agreed style: Inactive (No Indent), Active (Indent).
                # Transition?
                # If Triangle fades in/out, Text should slide?
                # If Text slides, use t.

                base_x = margin_left
                if i == target_idx or (target_idx > 0 and i == target_idx - 1):
                    # It's an active-ish line.
                    # Calculate indent based on tri_alpha logic implicitly
                    # Max indent = t_size + 10
                    max_indent = max(10, int(props["size"] * 0.5)) + 10

                    if i == target_idx:  # Incoming
                        current_indent = max_indent * t if t < 1.0 else max_indent
                    else:  # Outgoing
                        current_indent = max_indent * (1 - t) if t < 1.0 else 0

                    base_x += current_indent

                self.draw_multiline_text(
                    draw,
                    base_x,
                    props["y"],
                    lines,
                    this_font,
                    this_text_color,
                    stroke_width,
                    this_stroke_color,
                    this_shadow_color,
                    (shadow_offset_x, shadow_offset_y),
                    letter_spacing,
                    line_gap,
                )

            img_np = np.array(img).astype(np.float32) / 255.0
            del img
            del draw
            return img_np

        # PNG frames + FFmpeg for memory optimization
        print(f"LyricsScroll: Generating {total_frames} frames...")

        # Generate output filename with date and random number
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = random.randint(1000, 9999)
        output_filename = f"lyrics_{timestamp}_{random_suffix}.mov"

        # Ensure output directory exists
        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)

        video_path = os.path.join(output_dir, output_filename)
        print(f"LyricsScroll: Output path: {video_path}")

        # Create temporary directory for PNG frames
        # Try to use RAM disk (/dev/shm on Linux) for fastest I/O
        # Falls back to /tmp or system default if RAM disk not available
        temp_dir = None
        if os.path.exists("/dev/shm"):
            # Check available RAM disk space (at least 5GB needed for 2563 frames)
            try:
                stat = os.statvfs("/dev/shm")
                available_gb = stat.f_bavail * stat.f_frsize / (1024**3)
                if available_gb > 5:
                    temp_dir = "/dev/shm"
                    print(
                        f"LyricsScroll: Using RAM disk /dev/shm ({available_gb:.1f}GB available)"
                    )
            except:
                pass

        if temp_dir is None:
            temp_dir = "/tmp" if os.path.exists("/tmp") else None

        frames_dir = tempfile.mkdtemp(dir=temp_dir)
        print(f"LyricsScroll: Temp directory for frames: {frames_dir}")

        # Step 1: Generate all PNG frames
        print(f"LyricsScroll: Writing {total_frames} frames as PNG...")
        total_batches = (total_frames + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_frame = batch_idx * batch_size
            end_frame = min(start_frame + batch_size, total_frames)

            print(
                f"LyricsScroll: Processing batch {batch_idx + 1}/{total_batches} (frames {start_frame}-{end_frame - 1})..."
            )

            # Process each frame in this batch
            for f in range(start_frame, end_frame):
                frame_np = generate_single_frame(f)

                # Convert from RGBA (float32 [0,1]) to uint8 [0,255]
                frame_uint8 = (frame_np * 255).astype(np.uint8)

                # Save as PNG file
                img = Image.fromarray(frame_uint8, mode="RGBA")
                frame_path = os.path.join(frames_dir, f"frame_{f:06d}.png")
                img.save(frame_path)

                # Cleanup after each frame to minimize memory
                del frame_np, frame_uint8, img

            # Cleanup memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            print(f"LyricsScroll: Batch {batch_idx + 1}/{total_batches} completed")

        print(f"LyricsScroll: All PNG frames saved to {frames_dir}")

        # Debug: Check first PNG has transparency
        first_frame = os.path.join(frames_dir, "frame_000000.png")
        if os.path.exists(first_frame):
            test_img = Image.open(first_frame)
            print(
                f"LyricsScroll: First PNG info - Mode: {test_img.mode}, Size: {test_img.size}"
            )
            # Get pixel at (0,0) to verify transparency
            pixel = test_img.getpixel((0, 0))
            print(
                f"LyricsScroll: Pixel (0,0): R={pixel[0]} G={pixel[1]} B={pixel[2]} A={pixel[3]}"
            )
            if pixel[3] == 0:
                print("LyricsScroll: ✓ PNG has transparent alpha channel (A=0)")
            else:
                print(f"LyricsScroll: ✗ PNG alpha is not zero (A={pixel[3]})")

        # Step 2: Merge PNG frames into MOV video with ProRes codec (full alpha support)
        pattern = os.path.join(frames_dir, "frame_%06d.png")

        # Check if project-local FFmpeg 8 exists
        script_dir = os.path.dirname(os.path.abspath(__file__))
        local_ffmpeg = os.path.join(script_dir, "bin", "ffmpeg")

        if os.path.exists(local_ffmpeg):
            print(f"LyricsScroll: Using project-local FFmpeg: {local_ffmpeg}")
            ffmpeg_binary = local_ffmpeg
        else:
            print(
                "LyricsScroll: Using system FFmpeg (Note: FFmpeg 8+ recommended for alpha support)"
            )
            ffmpeg_binary = "ffmpeg"

        ffmpeg_cmd = [
            ffmpeg_binary,
            "-y",  # Overwrite output file
            "-framerate",
            str(frame_rate),
            "-i",
            pattern,
            "-c:v",
            "prores_ks",  # ProRes 4444 with full alpha support
            "-profile:v",
            "4",  # ProRes 4444 (唯一支持alpha的profile)
            "-pix_fmt",
            "yuva444p10le",  # YUV 4:4:4 with Alpha (标准格式)
            "-qscale:v",
            "5",  # Quality (1-22, 5 = high quality)
            video_path,
        ]

        print(f"LyricsScroll: Running FFmpeg to create video...")
        print(f"LyricsScroll: Command: {' '.join(ffmpeg_cmd)}")

        try:
            result = subprocess.run(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            print(f"LyricsScroll: FFmpeg completed successfully")

            # Verify output video pixel format with ffprobe
            try:
                ffprobe_cmd = [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=pix_fmt",
                    "-of",
                    "csv=p=0",
                    video_path,
                ]
                ffprobe_result = subprocess.run(
                    ffprobe_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if ffprobe_result.returncode == 0:
                    pix_fmt = ffprobe_result.stdout.strip()
                    print(f"LyricsScroll: Output video pixel format: {pix_fmt}")
                    if "yuva444p10le" in pix_fmt:
                        print("LyricsScroll: ✓ Video has alpha channel (yuva444p10le)")
                    else:
                        print(f"LyricsScroll: ✗ Video alpha channel missing: {pix_fmt}")
            except:
                print(
                    "LyricsScroll: Warning - ffprobe not available to verify pixel format"
                )

        except subprocess.CalledProcessError as e:
            print(f"LyricsScroll: FFmpeg failed with error:")
            print(f"  STDOUT: {e.stdout}")
            print(f"  STDERR: {e.stderr}")
            # Don't delete temp frames on failure for debugging
            print(f"LyricsScroll: Temp frames preserved at: {frames_dir}")
            raise RuntimeError(
                f"FFmpeg failed to create video. Check if ffmpeg is installed. "
                f"Temp frames saved at: {frames_dir}"
            ) from e

        # Step 3: Clean up temporary PNG frames
        try:
            shutil.rmtree(frames_dir)
            print(f"LyricsScroll: Cleaned temp directory {frames_dir}")
        except Exception as e:
            print(f"LyricsScroll: Warning - could not clean temp directory: {e}")

        print(f"LyricsScroll: Video saved to {video_path}")

        return (video_path, srt_text)


NODE_CLASS_MAPPINGS = {"LyricsScroll": LyricsScroll}

NODE_DISPLAY_NAME_MAPPINGS = {"LyricsScroll": "Lyrics Scroll Effect"}
