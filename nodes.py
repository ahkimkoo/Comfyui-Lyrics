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
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "margin_left": ("INT", {"default": 50, "min": 0, "max": 4096}),
                "margin_right": ("INT", {"default": 50, "min": 0, "max": 4096}),
                "y_pos": ("INT", {"default": 400, "min": 0, "max": 4096}),
                "font_size": ("INT", {"default": 30, "min": 10, "max": 200}), # Inactive size
                "active_font_size": ("INT", {"default": 40, "min": 10, "max": 200}), # Active size
                "letter_spacing": ("INT", {"default": 0, "min": -10, "max": 200}),
                "line_gap": ("INT", {"default": 20, "min": 0, "max": 1000}), # Gap between items
                "text_color": ("STRING", {"default": "#FFFFFF"}),
                "stroke_width": ("INT", {"default": 1, "min": 0, "max": 20}),
                "stroke_color": ("STRING", {"default": "#000000"}),
                "shadow_color": ("STRING", {"default": "#000000"}),
                "shadow_offset_x": ("INT", {"default": 2, "min": -100, "max": 100}),
                "shadow_offset_y": ("INT", {"default": 2, "min": -100, "max": 100}),
                "shadow_alpha": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.1}),
                "font_filename": (fonts,),
                "frame_rate": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "max_chars_per_line": ("INT", {"default": 20, "min": 5, "max": 100}),
                "whisper_prompt": (["简体中文", "繁体中文", "English", "日本語", "한국어"],),
            },
            "optional": {
                "audio": ("AUDIO",),
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
            if hasattr(font, 'getlength'):
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

    def draw_multiline_text(self, draw, x, y, lines, font, fill, stroke_width, stroke_fill, shadow_fill, shadow_offset, letter_spacing, line_gap):
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
            self.draw_line_custom(draw, x, current_y, line, font, fill, stroke_width, stroke_fill, shadow_fill, shadow_offset, letter_spacing)
            current_y += line_spacing

    def draw_line_custom(self, draw, x, y, text, font, fill, stroke_width, stroke_fill, shadow_fill, shadow_offset, letter_spacing):
        # Draw Shadow if enabled (alpha > 0)
        # shadow_fill is RGBA tuple.
        if shadow_fill and shadow_fill[3] > 0 and (shadow_offset[0] != 0 or shadow_offset[1] != 0):
            sx, sy = shadow_offset
            if letter_spacing == 0:
                draw.text((x + sx, y + sy), text, font=font, fill=shadow_fill, stroke_width=0)
            else:
                cursor_x = x
                for char in text:
                    if hasattr(font, 'getlength'): w = font.getlength(char)
                    else: w = font.size * 0.5 # fallback
                    
                    draw.text((cursor_x + sx, y + sy), char, font=font, fill=shadow_fill, stroke_width=0)
                    cursor_x += w + letter_spacing

        # Draw Main Text
        if letter_spacing == 0:
            if stroke_width > 0:
                draw.text((x, y), text, font=font, fill=stroke_fill, stroke_width=stroke_width, stroke_fill=stroke_fill)
            draw.text((x, y), text, font=font, fill=fill, stroke_width=0)
            return

        cursor_x = x
        for char in text:
            if hasattr(font, 'getlength'):
                w = font.getlength(char)
            else:
                try:
                    w, h = font.getsize(char)
                except:
                    w = font.size * 0.5

            if stroke_width > 0:
                 draw.text((cursor_x, y), char, font=font, fill=stroke_fill, stroke_width=stroke_width, stroke_fill=stroke_fill)
            draw.text((cursor_x, y), char, font=font, fill=fill, stroke_width=0)
            
            cursor_x += w + letter_spacing

    def parse_srt(self, srt_text):
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

    def transcribe_audio(self, audio_data, max_chars=20, prompt="简体中文"):
        if not WHISPER_AVAILABLE:
            raise ImportError("OpenAI Whisper is not installed. Please install it to generate subtitles automatically.")
        
        waveform = audio_data['waveform']
        sample_rate = audio_data['sample_rate']
        
        if waveform.dim() == 3:
            waveform = waveform[0]
        if waveform.dim() == 2:
            waveform = torch.mean(waveform, dim=0)
            
        if TORCHAUDIO_AVAILABLE and sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        elif sample_rate != 16000:
             raise ImportError("Torchaudio required for resampling.")
        
        print("LyricsScroll: Loading Whisper model...")
        model = whisper.load_model("base", device=self.device)
        print("LyricsScroll: Transcribing...")
        
        # Infer language logic or keep zh? User only asked for prompt.
        # But if prompt is English, language='zh' is bad.
        lang = "zh"
        if "English" in prompt: lang = "en"
        elif "日本" in prompt: lang = "ja"
        elif "한국" in prompt: lang = "ko"
        
        result = model.transcribe(waveform.numpy(), language=lang, initial_prompt=prompt)
        
        subs = []
        for segment in result['segments']:
            text = segment['text'].strip()
            start = segment['start']
            end = segment['end']
            
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
                    subs.append({'start': s, 'end': e, 'text': chunk})
            else:
                subs.append({'start': start, 'end': end, 'text': text})
        
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

    def generate_lyrics(self, width, height, margin_left, margin_right, y_pos, font_size, active_font_size, letter_spacing, line_gap, text_color, stroke_width, stroke_color, shadow_color, shadow_offset_x, shadow_offset_y, shadow_alpha, font_filename, frame_rate, max_chars_per_line, whisper_prompt, audio=None, text=""):
        
        self.font_cache = {}

        subs = []
        srt_text = ""

        # 1. Try to use provided text
        if text and text.strip():
            print("LyricsScroll: Using provided text/SRT.")
            subs = self.parse_srt(text)
            srt_text = text
            if not subs:
                print("LyricsScroll: Warning - Provided text is not valid SRT.")
        
        # 2. If no text, try Whisper (requires audio)
        if not subs:
            if audio is not None:
                print(f"LyricsScroll: No text provided. Running Whisper with max_chars={max_chars_per_line}, prompt='{whisper_prompt}'...")
                subs, srt_text = self.transcribe_audio(audio, max_chars=max_chars_per_line, prompt=whisper_prompt)
            else:
                raise ValueError("LyricsScroll: Either 'audio' or 'text' (SRT) must be provided. If providing text only, ensure it is valid SRT format.")

        # 3. Determine Duration
        if audio is not None:
            waveform = audio['waveform']
            sample_rate = audio['sample_rate']
            total_samples = waveform.shape[-1]
            duration = total_samples / sample_rate
        elif subs:
            # Fallback duration from subtitles if no audio
            duration = subs[-1]['end'] + 2.0 # Add padding
        else:
            duration = 10.0 # Should not happen
            
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
        
        print(f"LyricsScroll: Generating {total_frames} frames with slot-based animation...")
        
        if not subs:
            empty = torch.zeros((max(1, total_frames), height, width, 4), dtype=torch.float32)
            return (empty, srt_text)

        ANIMATION_DURATION = 0.3 # 300ms
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
                _, _, h = self.wrap_text(subs[idx]['text'], font, MAX_TEXT_WIDTH, letter_spacing, line_gap)
                return h

            # Central Item (Slot 2)
            h_center = get_h(item_idx, active_font_size)
            y_center = y_pos - h_center / 2
            
            # If item_idx is valid, it's visible opacity 255. If virtual (-1), 0.
            alpha_center = 255 if 0 <= item_idx < len(subs) else 0
            
            layout[item_idx] = {'y': y_center, 'size': active_font_size, 'alpha': alpha_center}
            
            # Top Item (Slot 1) - Item N-1
            idx_top = item_idx - 1
            if idx_top >= 0:
                h_top = get_h(idx_top, font_size)
                # Position: Above center item.
                # y_top = y_center - GAP - h_top
                y_top = y_center - GAP - h_top
                layout[idx_top] = {'y': y_top, 'size': font_size, 'alpha': 150}
                
            # Bottom Item (Slot 3) - Item N+1
            idx_bot = item_idx + 1
            if idx_bot < len(subs):
                h_bot = get_h(idx_bot, font_size)
                # Position: Below center item
                y_bot = y_center + h_center + GAP
                layout[idx_bot] = {'y': y_bot, 'size': font_size, 'alpha': 150}
                
            # "Out" positions for transitions
            # Item N-2 (Top Out)
            idx_out_top = item_idx - 2
            if idx_out_top >= 0:
                h_out = get_h(idx_out_top, font_size)
                # Even further up
                # Need reference to y_top
                if idx_top >= 0:
                    y_out = layout[idx_top]['y'] - GAP - h_out
                else:
                    y_out = y_center - GAP - h_out # Fallback
                layout[idx_out_top] = {'y': y_out, 'size': font_size, 'alpha': 0}

            # Item N+2 (Bottom In)
            idx_in_bot = item_idx + 2
            if idx_in_bot < len(subs):
                # Start further down
                if idx_bot < len(subs):
                    y_in = layout[idx_bot]['y'] + get_h(idx_bot, font_size) + GAP
                else:
                    y_in = y_center + h_center + GAP
                layout[idx_in_bot] = {'y': y_in, 'size': font_size, 'alpha': 0}
                
            return layout

        for f in range(total_frames):
            current_time = f / frame_rate
            
            # 1. Identify the "Next" line to determine phase
            # next_idx is the first subtitle that hasn't started yet (start > current_time)
            next_idx = len(subs)
            for i, sub in enumerate(subs):
                if sub['start'] > current_time:
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
            def ease_out_cubic(t): return 1 - math.pow(1 - t, 3)
            
            is_transitioning = False
            
            if next_idx < len(subs):
                next_start = subs[next_idx]['start']
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
                        p1 = layout_prev.get(k, {'y': y_pos + height/2, 'size': font_size, 'alpha': 0}) 
                        p2 = layout_next.get(k, {'y': y_pos - height/2, 'size': font_size, 'alpha': 0})
                        
                        curr_y = p1['y'] + (p2['y'] - p1['y']) * t
                        curr_size = p1['size'] + (p2['size'] - p1['size']) * t
                        curr_alpha = p1['alpha'] + (p2['alpha'] - p1['alpha']) * t
                        
                        layout_current[k] = {'y': curr_y, 'size': curr_size, 'alpha': curr_alpha}
            
            if not is_transitioning:
                # Settled at current_state_idx
                layout_current = calculate_layout(current_state_idx, True)
                target_idx = current_state_idx
                t = 1.0
                
            # Render
            img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Sort by index to draw in order
            sorted_indices = sorted(layout_current.keys())
            
            for i in sorted_indices:
                props = layout_current[i]
                if props['alpha'] < 1: continue
                
                this_font = self.get_font(font_filename, int(props['size']))
                
                # Colors
                alpha_int = int(props['alpha'])
                this_text_color = text_color_rgb + (alpha_int,)
                this_stroke_color = stroke_color_rgb + (alpha_int,)
                
                # Shadow Color with combined alpha
                # Shadow base alpha * Text Fade alpha
                this_shadow_alpha = int(255 * shadow_alpha * (alpha_int / 255.0))
                this_shadow_color = shadow_color_rgb + (this_shadow_alpha,)
                
                # Text Wrapping
                lines, _, _ = self.wrap_text(subs[i]['text'], this_font, MAX_TEXT_WIDTH, letter_spacing, line_gap)
                
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
                    if t < 1.0: # Transitioning
                         tri_alpha = int(255 * t)
                    else:
                         tri_alpha = 255
                elif target_idx > 0 and i == target_idx - 1:
                    # Outgoing focus
                    # If transitioning, 255 -> 0
                    if t < 1.0:
                        tri_alpha = int(255 * (1-t))
                
                # Draw Triangle
                if tri_alpha > 10:
                    t_size = max(10, int(props['size'] * 0.5))
                    tri_x = margin_left
                    # Center Y relative to first line of text?
                    # Text Y is top-left.
                    # Calculate single line height for centering triangle on first line
                    ascent, descent = this_font.getmetrics()
                    first_line_h = ascent + descent
                    tri_y = props['y'] + first_line_h / 2
                    
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
                    max_indent = max(10, int(props['size'] * 0.5)) + 10
                    
                    if i == target_idx: # Incoming
                        current_indent = max_indent * t if t < 1.0 else max_indent
                    else: # Outgoing
                        current_indent = max_indent * (1-t) if t < 1.0 else 0
                        
                    base_x += current_indent
                
                self.draw_multiline_text(draw, base_x, props['y'], lines, this_font, this_text_color, stroke_width, this_stroke_color, this_shadow_color, (shadow_offset_x, shadow_offset_y), letter_spacing, line_gap)

            img_np = np.array(img).astype(np.float32) / 255.0
            output_images.append(torch.from_numpy(img_np))
            
        return (torch.stack(output_images), srt_text)

NODE_CLASS_MAPPINGS = {
    "LyricsScroll": LyricsScroll
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LyricsScroll": "Lyrics Scroll Effect"
}
