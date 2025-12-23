import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor
import folder_paths
import os
import re
import math

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

    @classmethod
    def INPUT_TYPES(s):
        # Ensure fonts directory exists and list fonts
        font_dir = os.path.join(folder_paths.models_dir, "fonts")
        os.makedirs(font_dir, exist_ok=True)
        fonts = [f for f in os.listdir(font_dir) if f.lower().endswith(('.ttf', '.otf'))]
        if not fonts:
            fonts = ["Arial.ttf"] # Fallback

        return {
            "required": {
                "audio": ("AUDIO",),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "x_pos": ("INT", {"default": 50, "min": 0, "max": 4096}),
                "y_pos": ("INT", {"default": 400, "min": 0, "max": 4096}),
                "font_size": ("INT", {"default": 30, "min": 10, "max": 200}),
                "active_font_size": ("INT", {"default": 40, "min": 10, "max": 200}),
                "letter_spacing": ("INT", {"default": 15, "min": -10, "max": 200}),
                "text_color": ("STRING", {"default": "#FFFFFF"}),
                "stroke_width": ("INT", {"default": 2, "min": 0, "max": 20}),
                "stroke_color": ("STRING", {"default": "#000000"}),
                "font_filename": (fonts,),
                "frame_rate": ("INT", {"default": 25, "min": 1, "max": 120}),
                "max_chars_per_line": ("INT", {"default": 20, "min": 5, "max": 100}),
                "visible_lines": ("INT", {"default": 3, "min": 1, "max": 20}),
                "active_line_index": ("INT", {"default": 2, "min": 1, "max": 20}),
            },
            "optional": {
                "text": ("STRING", {"multiline": True, "default": "", "forceInput": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "subtitles")
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

    def draw_text_custom(self, draw, x, y, text, font, fill, stroke_width, stroke_fill, letter_spacing):
        if letter_spacing == 0:
            if stroke_width > 0:
                draw.text((x, y), text, font=font, fill=stroke_fill, stroke_width=stroke_width, stroke_fill=stroke_fill)
            draw.text((x, y), text, font=font, fill=fill, stroke_width=0)
            return

        cursor_x = x
        for char in text:
            # Get char width compatibility
            if hasattr(font, 'getlength'):
                w = font.getlength(char)
            else:
                try:
                    w, h = font.getsize(char)
                except:
                    w = font_size * 0.5 # fallback

            if stroke_width > 0:
                 draw.text((cursor_x, y), char, font=font, fill=stroke_fill, stroke_width=stroke_width, stroke_fill=stroke_fill)
            draw.text((cursor_x, y), char, font=font, fill=fill, stroke_width=0)
            
            cursor_x += w + letter_spacing

    def parse_srt(self, srt_text):
        """Parses SRT string into a list of (start_seconds, end_seconds, text)."""
        pattern = re.compile(r'(\d+)\n(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})\n(.*?)(?=\n\n|\n$|$)', re.DOTALL)
        matches = pattern.findall(srt_text.replace('\r\n', '\n') + '\n\n')
        
        subs = []
        for match in matches:
            idx, h1, m1, s1, ms1, h2, m2, s2, ms2, content = match
            start = int(h1)*3600 + int(m1)*60 + int(s1) + int(ms1)/1000
            end = int(h2)*3600 + int(m2)*60 + int(s2) + int(ms2)/1000
            content = content.strip()
            subs.append({'start': start, 'end': end, 'text': content})
        return subs

    def transcribe_audio(self, audio_data, max_chars=20):
        if not WHISPER_AVAILABLE:
            raise ImportError("OpenAI Whisper is not installed. Please install it to generate subtitles automatically.")
        
        # Audio is usually {'waveform': tensor[B, C, N], 'sample_rate': int}
        waveform = audio_data['waveform']
        sample_rate = audio_data['sample_rate']
        
        # Handle batch (take first) and channels (mix to mono)
        if waveform.dim() == 3:
            waveform = waveform[0] # Take first batch
        if waveform.dim() == 2:
            waveform = torch.mean(waveform, dim=0) # Mix channels
            
        # Resample to 16000 for Whisper
        if TORCHAUDIO_AVAILABLE and sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        elif sample_rate != 16000:
             raise ImportError("Torchaudio required for resampling.")
        
        print("LyricsScroll: Loading Whisper model...")
        model = whisper.load_model("base", device=self.device)
        print("LyricsScroll: Transcribing...")
        result = model.transcribe(waveform.numpy(), language="zh")
        
        subs = []
        
        # Process and split segments
        for segment in result['segments']:
            text = segment['text'].strip()
            start = segment['start']
            end = segment['end']
            
            if len(text) > max_chars:
                # Split logic
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
                    subs.append({'start': s, 'end': e, 'text': chunk})
            else:
                subs.append({'start': start, 'end': end, 'text': text})
        
        # Re-generate SRT string
        srt_output = ""
        for i, sub in enumerate(subs):
            s = sub['start']
            e = sub['end']
            txt = sub['text']
            
            s_h, s_r = divmod(s, 3600)
            s_m, s_s = divmod(s_r, 60)
            s_ms = (s_s - int(s_s)) * 1000
            
            e_h, e_r = divmod(e, 3600)
            e_m, e_s = divmod(e_r, 60)
            e_ms = (e_s - int(e_s)) * 1000
            
            srt_output += f"{i+1}\n{int(s_h):02}:{int(s_m):02}:{int(s_s):02},{int(s_ms):03} --> {int(e_h):02}:{int(e_m):02}:{int(e_s):02},{int(e_ms):03}\n{txt}\n\n"
            
        return subs, srt_output

    def generate_lyrics(self, audio, width, height, x_pos, y_pos, font_size, active_font_size, letter_spacing, text_color, stroke_width, stroke_color, font_filename, frame_rate, max_chars_per_line, visible_lines, active_line_index, text=""):
        
        # Clear font cache at start of generation
        self.font_cache = {}

        # 1. Prepare Subtitles
        if text and text.strip():
            print("LyricsScroll: Using provided text/SRT.")
            subs = self.parse_srt(text)
            srt_text = text
            if not subs:
                print("LyricsScroll: Warning - Provided text is not valid SRT.")
        else:
            print(f"LyricsScroll: No text provided. Running Whisper with max_chars={max_chars_per_line}...")
            subs, srt_text = self.transcribe_audio(audio, max_chars=max_chars_per_line)
        
        # 2. Audio Duration
        waveform = audio['waveform']
        sample_rate = audio['sample_rate']
        total_samples = waveform.shape[-1]
        duration = total_samples / sample_rate
        total_frames = int(duration * frame_rate)
        
        # 3. Configuration
        line_height = int(active_font_size * 1.5) # Use active font size for line height stability? Or normal? 
        # Usually list spacing should handle the largest font.
        
        try:
            text_color_rgb = ImageColor.getrgb(text_color)
            stroke_color_rgb = ImageColor.getrgb(stroke_color)
        except:
            text_color_rgb = (255, 255, 255)
            stroke_color_rgb = (0, 0, 0)
            
        icon_size_base = int(active_font_size * 0.4) # Smaller triangle
        
        output_images = []
        
        print(f"LyricsScroll: Generating {total_frames} frames with scrolling animation...")
        
        if not subs:
            print("LyricsScroll: No subtitles found. Returning blank frames.")
            empty = torch.zeros((max(1, total_frames), height, width, 4), dtype=torch.float32)
            return (empty, srt_text)

        # Convert 1-based index to 0-based
        focus_idx_0 = max(0, min(active_line_index - 1, visible_lines - 1))
        
        ANIMATION_DURATION = 0.8 # Scroll animation duration (max)
        ZOOM_DURATION = 0.3      # Zoom animation duration
        
        for f in range(total_frames):
            current_time = f / frame_rate
            
            # Find integer target index
            target_idx_int = -1
            for i, sub in enumerate(subs):
                if sub['start'] <= current_time < sub['end']:
                    target_idx_int = i
                    break
            
            # If in gap, find next
            if target_idx_int == -1:
                for i, sub in enumerate(subs):
                    if sub['start'] > current_time:
                        target_idx_int = i 
                        break
                if target_idx_int == -1:
                    target_idx_int = len(subs) - 1
            
            target_idx_int = max(0, min(target_idx_int, len(subs)-1))
            
            # Calculate Fractional Target (Scroll Animation)
            fractional_target = float(target_idx_int)
            
            if target_idx_int > 0:
                curr_start = subs[target_idx_int]['start']
                prev_start = subs[target_idx_int - 1]['start']
                gap = curr_start - prev_start
                
                # Dynamic duration: ensure we don't start animating before previous line starts
                # Using min(ANIMATION_DURATION, gap * 0.8) to be safe
                this_anim_dur = min(ANIMATION_DURATION, gap * 0.9)
                if this_anim_dur < 0.1: this_anim_dur = 0.1 # Minimum safety

                if curr_start - this_anim_dur <= current_time < curr_start:
                    progress = (current_time - (curr_start - this_anim_dur)) / this_anim_dur
                    progress = max(0.0, min(1.0, progress))
                    # Ease In Out Cubic for Scroll (Smoother than SmoothStep)
                    # t < 0.5 ? 4 * t * t * t : 1 - pow(-2 * t + 2, 3) / 2
                    if progress < 0.5:
                        ease = 4 * progress * progress * progress
                    else:
                        ease = 1 - math.pow(-2 * progress + 2, 3) / 2
                    
                    fractional_target = (target_idx_int - 1) + ease
            
            # Calculate Window Start (Float)
            window_start_float = fractional_target - focus_idx_0
            
            # Clamp Window
            if window_start_float < 0:
                window_start_float = 0.0
            
            if len(subs) >= visible_lines:
                max_start = len(subs) - visible_lines
                if window_start_float > max_start:
                    window_start_float = float(max_start)
            else:
                window_start_float = 0.0

            # Create transparent image
            img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            start_render_idx = int(math.floor(window_start_float))
            # Fix: Use ceil to strictly control visible lines count. 
            # Static state -> visible_lines. Scrolling -> visible_lines + 1.
            end_render_idx = int(math.ceil(window_start_float + visible_lines))
            
            base_y_offset = window_start_float + focus_idx_0
            
            for i in range(start_render_idx, end_render_idx):
                if i < 0 or i >= len(subs):
                    continue
                    
                sub = subs[i]
                
                # Calculate Y
                line_offset = i - base_y_offset
                draw_y = y_pos + line_offset * line_height
                
                # Zoom / Alpha Logic
                this_font_size = font_size
                alpha = 150
                is_active = (i == target_idx_int)
                
                if is_active:
                    # Active Line
                    alpha = 255
                    time_in_state = current_time - sub['start']
                    if time_in_state < 0: time_in_state = 0
                    
                    if time_in_state < ZOOM_DURATION:
                        linear_p = time_in_state / ZOOM_DURATION
                        # Ease Out Cubic: 1 - (1-t)^3 (Fast start, slow end)
                        zoom_p = 1.0 - math.pow(1.0 - linear_p, 3)
                        this_font_size = font_size + (active_font_size - font_size) * zoom_p
                    else:
                        this_font_size = active_font_size
                
                # Dynamic Font
                this_font = self.get_font(font_filename, int(this_font_size))
                
                # Colors
                this_text_color = text_color_rgb + (int(alpha),)
                this_stroke_color_rgba = stroke_color_rgb + (int(alpha),)

                # Center text vertically based on line_height?
                offset_y_center = (line_height - int(this_font_size * 1.5)) / 2
                
                # Triangle Settings
                t_size = max(10, int(active_font_size * 0.5))
                
                # Position Logic
                # Active Line: Triangle at x_pos. Text at x_pos + triangle_width.
                # Inactive Line: Text at x_pos.
                
                current_text_x = x_pos
                
                if is_active:
                    # Draw Triangle
                    # Triangle Position: Left aligned at x_pos
                    tri_left_x = x_pos
                    tri_center_y = draw_y + line_height / 2
                    
                    # Shape "‣"
                    p1 = (tri_left_x + t_size, tri_center_y) # Tip
                    p2 = (tri_left_x, tri_center_y - t_size * 0.6) # Back Top
                    p3 = (tri_left_x, tri_center_y + t_size * 0.6) # Back Bottom
                    
                    # Ensure opacity is full for the indicator
                    draw.polygon([p1, p2, p3], fill=text_color_rgb + (255,), outline=stroke_color_rgb + (255,))
                    
                    # Shift text
                    current_text_x = x_pos + t_size + 10 # 10px padding
                
                # Render text with letter spacing
                self.draw_text_custom(draw, current_text_x, draw_y + offset_y_center, sub['text'], this_font, this_text_color, stroke_width, this_stroke_color_rgba, letter_spacing)
            
            img_np = np.array(img).astype(np.float32) / 255.0
            output_images.append(torch.from_numpy(img_np))
            
        return (torch.stack(output_images), srt_text)

NODE_CLASS_MAPPINGS = {
    "LyricsScroll": LyricsScroll
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LyricsScroll": "Lyrics Scroll Effect"
}
