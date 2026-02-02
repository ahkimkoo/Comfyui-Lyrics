# Alpha Channel Support Guide

## Overview

The `Comfyui-Lyrics` project has been refactored to generate RGBA image sequences with transparency support. This guide explains how to save videos with alpha channels.

## Current Architecture

### Main Changes in New Version

**Before (Old Version)**:
- Node directly encoded video using FFmpeg with ProRes 4444
- Output: `.mov` file with alpha channel
- Pixel format: `yuva444p12le`

**After (Current Version)**:
- Node outputs RGBA image tensor (IMAGE format)
- Requires separate video save node
- More flexible: supports multiple video formats

### Lyrics Scroll Node

- **Input**: Audio or SRT text
- **Output**: RGBA image sequence with transparent background
- **Category**: `Lyrics`
- **Format**: ComfyUI IMAGE tensor (batch, height, width, 4 channels)

## How to Save Video with Alpha Channel

### Method 1: Using Video Save (Alpha Support) Node (Recommended)

A new node `Video Save (Alpha Support)` has been added to save videos with alpha channels.

**Usage**:
1. Add `Lyrics Scroll Effect` node
2. Add `Video Save (Alpha Support)` node
3. Connect `images` output from Lyrics Scroll to `images` input of Video Save
4. Select video format:
   - `prores_ks` - ProRes 4444 (best quality, .mov)
   - `libvpx-vp9` - VP9 with alpha (.webm)
   - `qtrle` - QuickTime Animation (lossless, .mov)
   - `png` - PNG sequence (no encoding)

**Node Parameters**:
- `images`: Input RGBA image sequence
- `frame_rate`: Video frame rate (default: 25.0)
- `filename_prefix`: Output filename prefix
- `format`: Video codec (see options above)
- `quality`: Quality setting (1-31 for ProRes)

**Expected Output**:
```
VideoSaveWithAlpha: Input images have alpha channel (RGBA: 1000 frames, 720x1280)
VideoSaveWithAlpha: Using ProRes 4444 with alpha channel
VideoSaveWithAlpha: FFmpeg completed successfully
VideoSaveWithAlpha: Output video pixel format: yuva444p12le
✓ Video has alpha channel (yuva444p12le)
```

### Method 2: Using VHS_VideoCombine (ComfyUI Default)

If you have the [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) installed:

1. Add `VHS_VideoCombine` node
2. Connect `images` from Lyrics Scroll to `images` input
3. Configure codec for alpha support:
   - **Format**: `video/quicktime` or `video/webm`
   - **Codec**:
     - `prores_ks` (ProRes 4444) → Set profile to 4
     - `libvpx-vp9` (VP9) → Enable alpha
   - **PixelFormat**: `yuva444p10le` or `yuva444p12le`

**Note**: VHS_VideoCombine may not support all alpha formats out of the box. The custom `Video Save (Alpha Support)` node is recommended for best alpha support.

### Method 3: Manual FFmpeg Export (Advanced)

If you need precise control over encoding parameters:

1. Save image sequence using a node like `Save Image` (with filename prefix like `frame_`)
2. Use FFmpeg command line:

```bash
# ProRes 4444 with alpha
ffmpeg -framerate 25 -i frame_%06d.png -c:v prores_ks -profile:v 4 -pix_fmt yuva444p10le -qscale:v 5 output.mov

# VP9 with alpha (WebM)
ffmpeg -framerate 25 -i frame_%06d.png -c:v libvpx-vp9 -pix_fmt yuva420p -crf 23 output.webm
```

## Video Format Comparison

| Format | Extension | Alpha Support | Quality | File Size | Software Compatibility |
|--------|-----------|--------------|----------|-----------|---------------------|
| **ProRes 4444** | .mov | ✓ Yes (yuva444p*) | High | Medium | DaVinci Resolve, FCPX, Premiere |
| **VP9** | .webm | ✓ Yes (yuva420p) | Medium | Small | Browsers, DaVinci Resolve |
| **QuickTime Animation** | .mov | ✓ Yes | Lossless | Large | QuickTime, FCPX |
| **H.264** | .mp4 | ✗ No | High | Small | Universal |

## Alpha Channel Verification

### Verify with ffprobe

```bash
# Check pixel format
ffprobe -v error -select_streams v:0 -show_entries stream=pix_fmt -of csv=p=0 video.mov

# Expected output for alpha: yuva444p10le, yuva444p12le, yuva420p (with 'a')
# Expected output without alpha: yuv420p, rgb24
```

### Verify with Video Player

- **DaVinci Resolve**: Import video → Check color space → Should show "Alpha"
- **QuickTime Player**: Does NOT display transparency (bug/limitation)
- **VLC**: Does NOT display transparency
- **Web Browsers**: VP9/WebM with alpha works, ProRes may need transcoding

### Extract Frame and Check Alpha

```bash
# Extract first frame
ffmpeg -i video.mov -vframes 1 -update 1 frame.png

# Check with Python
python -c "
from PIL import Image
img = Image.open('frame.png')
print(f'Mode: {img.mode}')
print(f'Has alpha: {img.mode in (\"RGBA\", \"LA\", \"PA\")}')
if img.mode == 'RGBA':
    alpha = img.split()[-1]
    print(f'Alpha range: {min(alpha.getdata())} - {max(alpha.getdata())}')
"
```

## Common Issues

### Issue: "Video alpha channel missing"

**Symptoms**:
- Log shows `yuva444p12le` but reports alpha missing
- Code only checks for `yuva444p10le`

**Root Cause**:
- ProRes 4444 encoder prefers 12-bit (`yuva444p12le`) over 10-bit
- Old validation code only checked for `yuva444p10le`

**Solution**:
- Update validation to accept any `yuva444p*` format
- Use `Video Save (Alpha Support)` node (already fixed)

### Issue: Black background in exported video

**Symptoms**:
- Video has black background instead of transparency
- Alpha channel present but all pixels are opaque

**Root Cause**:
- Input images may not have alpha channel
- Encoder converted RGBA to RGB

**Solution**:
1. Verify Lyrics Scroll output is RGBA (not RGB)
2. Check PNG frames have alpha < 255
3. Use codec that supports alpha (ProRes 4444, VP9)

### Issue: QuickTime shows solid background

**Symptoms**:
- Video has alpha channel (verified with ffprobe)
- QuickTime Player shows black/solid background

**Root Cause**:
- QuickTime Player does not display alpha channels (known limitation)

**Solution**:
- Test in DaVinci Resolve or professional software
- Use VP9/WebM format for web compatibility

## Workflow Example

### Complete Alpha Channel Workflow

```
1. Load Audio [AudioLoad]
   ↓
2. Lyrics Scroll Effect (generates RGBA frames)
   ↓
3. Video Save (Alpha Support)
   - Format: prores_ks
   - Frame rate: 25.0
   - Quality: 5
   ↓
4. Output: lyrics_video_1234.mov with transparent background
```

### Compose with Background Video

If you want to overlay lyrics on a background video:

**Option A: Post-Production (Recommended)**
1. Export lyrics video with alpha
2. Import both videos into DaVinci Resolve / Premiere
3. Use compositing tool to layer videos

**Option B: In ComfyUI**
1. Generate background video separately
2. Use `ImageBlend` or `LayerUtility` node to composite
3. Note: This requires custom nodes

## Reference

- [FFmpeg ProRes Encoding](https://ffmpeg.org/ffmpeg-codecs.html#prores_ks)
- [ProRes 4444 Specifications](https://support.apple.com/kb/HT4186)
- [ComfyUI VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)
- [VP9 Alpha Support](https://www.webmproject.org/docs/encoder-parameters/)

## FAQ

**Q: Why use ProRes 4444 instead of H.264?**
A: H.264 does not support alpha channels. ProRes 4444 is the industry standard for video with transparency.

**Q: Can I use H.265 with alpha?**
A: No, H.265/HEVC does not support alpha channels.

**Q: What's the difference between yuva444p10le and yuva444p12le?**
A: Both support alpha. 10-bit = 1024 color levels per channel, 12-bit = 4096 levels. FFmpeg prefers 12-bit for ProRes 4444.

**Q: Which format should I use for web?**
A: VP9/WebM with alpha is supported by modern browsers. For maximum compatibility, use PNG sequence with JavaScript player.

**Q: Is the alpha channel lossless?**
A: ProRes 4444 and QuickTime Animation are visually lossless. VP9 is lossy but can achieve high quality with low CRF values.
