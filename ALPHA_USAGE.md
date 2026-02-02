# Alpha Channel Usage Guide

## Quick Summary

**Main branch has been refactored!** The node now outputs RGBA image sequences instead of directly encoding video files.

## Changes from Old Version

| Feature | Old Version | New Version |
|---------|-------------|-------------|
| Output | `.mov` file | RGBA IMAGE tensor |
| Alpha Channel | Embedded in video | Embedded in image sequence |
| Video Encoding | Built-in FFmpeg | Requires separate save node |
| Pixel Format Check | `yuva444p10le` | Not applicable |

## How to Use

### Step 1: Generate Lyrics (with Alpha)

Add the **"Lyrics Scroll Effect"** node:
- Category: `Lyrics`
- Input: Audio or SRT text
- Output: RGBA image sequence with **transparent background**

The node now generates images where the background is fully transparent `(0, 0, 0, 0)`.

### Step 2: Save as Video with Alpha

#### Option A: Use the New "Video Save (Alpha Support)" Node (Recommended)

A new node has been added specifically for saving videos with alpha channels:

```
Node: Video Save (Alpha Support)
Category: Lyrics
```

**Parameters**:
- `images`: Connect from Lyrics Scroll node
- `frame_rate`: Video frame rate (default: 25.0)
- `filename_prefix`: Output filename (default: "lyrics_video")
- `format`: Video codec
  - `prores_ks` - ProRes 4444 (best quality, .mov, recommended)
  - `libvpx-vp9` - VP9 with alpha (.webm, web-compatible)
  - `qtrle` - QuickTime Animation (lossless, .mov)
  - `png` - PNG sequence (no encoding)
- `quality`: Quality (1-22 for ProRes, 1-63 for VP9)

**Expected Output**:
```
✓ Input images have alpha channel (RGBA: 1000 frames, 720x1280)
✓ Using ProRes 4444 with alpha channel
✓ FFmpeg completed successfully
✓ Output video pixel format: yuva444p12le
✓ Video has alpha channel (yuva444p12le)
```

#### Option B: Use VHS_VideoCombine (ComfyUI Standard)

If you have [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite):

1. Add `VHS_VideoCombine` node
2. Connect Lyrics Scroll `images` → VHS `images`
3. Configure:
   - Format: `video/quicktime` or `video/webm`
   - Codec: `prores_ks`
   - PixelFormat: `yuva444p10le`
   - For ProRes 4444, set profile to 4

**Note**: Custom `Video Save (Alpha Support)` node is recommended for better alpha support.

## Video Format Recommendations

| Format | Extension | When to Use |
|--------|-----------|-------------|
| **ProRes 4444** | .mov | Professional editing, best quality |
| **VP9** | .webm | Web browsers, smaller file size |
| **QuickTime Animation** | .mov | Lossless, larger files |
| **PNG Sequence** | .png | Maximum control, no compression |

## Verify Alpha Channel

### Check with ffprobe:
```bash
ffprobe -v error -select_streams v:0 -show_entries stream=pix_fmt -of csv=p=0 video.mov
```

**Expected output**: `yuva444p12le`, `yuva444p10le`, or any format containing "a" (for alpha)

### Test in Software:
- ✓ **DaVinci Resolve**: Shows alpha correctly
- ✓ **Final Cut Pro**: Shows alpha correctly
- ✓ **Adobe Premiere**: Shows alpha correctly
- ✗ **QuickTime Player**: Does not show alpha (known limitation)
- ✗ **VLC**: Does not show alpha (known limitation)

## FAQ

**Q: I see "Video alpha channel missing: yuva444p12le" in logs**
A: This was an old bug in the previous version. The new version outputs images, so this error doesn't apply anymore.

**Q: Why is the video format yuva444p12le instead of yuva444p10le?**
A: FFmpeg's ProRes encoder prefers 12-bit color depth. Both formats support alpha channels.

**Q: The video has a black background**
A: Make sure you're using a format that supports alpha (ProRes 4444, VP9). Test in professional software like DaVinci Resolve, not QuickTime Player.

**Q: Can I use H.264 with alpha?**
A: No, H.264 does not support alpha channels. Use ProRes 4444, VP9, or QuickTime Animation.

**Q: Which format is best for web?**
A: VP9/WebM with alpha is supported by modern browsers. For maximum compatibility, export as PNG sequence and use JavaScript.

## Example Workflow

```
[AudioLoad] → [Lyrics Scroll Effect] → [Video Save (Alpha Support)]
                                                  ↓
                                         lyrics_video_1234.mov
                                         (with transparent background)
```

## Files Added

1. **video_save_node.py** - New node for saving videos with alpha channel support
2. **ALPHA_GUIDE.md** - Comprehensive alpha channel documentation
3. **ALPHA_USAGE.md** - This quick start guide (you're reading it)

## For More Details

See `ALPHA_GUIDE.md` for:
- Detailed format comparison
- FFmpeg command examples
- Troubleshooting common issues
- Advanced workflows
