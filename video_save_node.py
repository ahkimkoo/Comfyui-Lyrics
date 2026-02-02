"""
ComfyUI Video Save Node with Alpha Channel Support
支持 Alpha 通道的视频保存节点（ProRes 4444）
"""

import torch
import numpy as np
import subprocess
import os
import shutil
import folder_paths
from pathlib import Path


class VideoSaveWithAlpha:
    """Save image sequence as video with alpha channel support"""

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.temp_dir = folder_paths.get_temp_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 120.0}),
                "filename_prefix": ("STRING", {"default": "lyrics_video"}),
                "format": (
                    ["prores_ks", "libvpx-vp9", "qtrle", "png"],
                    {"default": "prores_ks"},
                ),
            },
            "optional": {
                "force_size": ("BOOLEAN", {"default": False}),
                "quality": ("INT", {"default": 5, "min": 1, "max": 31}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "save_video"
    CATEGORY = "Lyrics"
    OUTPUT_NODE = True

    def save_video(
        self, images, frame_rate, filename_prefix, format, force_size=False, quality=5
    ):
        """
        Save image sequence as video with alpha channel

        Args:
            images: Image tensor (B, H, W, C) - should be RGBA format
            frame_rate: Video frame rate
            filename_prefix: Output filename prefix
            format: Video format (prores_ks for ProRes 4444 with alpha)
            force_size: Force specific size settings
            quality: Video quality (1-31 for VP9, 1-22 for ProRes)

        Returns:
            video_path: Path to saved video file
        """

        # Check if images have alpha channel
        batch_size, height, width, channels = images.shape
        if channels < 4:
            print("Warning: Input images do not have alpha channel (channels < 4)")
            # Convert RGB to RGBA
            alpha_channel = torch.ones(
                (batch_size, height, width, 1), device=images.device
            )
            images = torch.cat([images, alpha_channel], dim=3)
        elif channels == 4:
            print(
                f"Input images have alpha channel (RGBA: {batch_size} frames, {width}x{height})"
            )
        else:
            print(f"Warning: Unexpected channel count: {channels}")

        # Generate output filename
        output_filename = f"{filename_prefix}_{''.join(map(str, [int(x) for x in torch.rand(4) * 10]))}.{self._get_extension(format)}"

        # Use RAM disk if available for better performance
        use_ramdisk = self._check_ram_disk()
        if use_ramdisk:
            import tempfile

            frames_dir = tempfile.mkdtemp(prefix="frames_", dir="/dev/shm")
        else:
            frames_dir = folder_paths.get_temp_directory()

        print(f"VideoSaveWithAlpha: Saving {batch_size} frames to {frames_dir}")

        try:
            # Step 1: Save frames as PNG files
            print(f"VideoSaveWithAlpha: Writing {batch_size} frames as PNG...")
            for i, img in enumerate(images):
                img_np = (img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

                # Convert RGB(A) to PIL Image
                from PIL import Image

                if channels >= 4:
                    img_pil = Image.fromarray(img_np, "RGBA")
                else:
                    img_pil = Image.fromarray(img_np, "RGB")

                frame_path = os.path.join(frames_dir, f"frame_{i:06d}.png")
                img_pil.save(frame_path, "PNG")

            print(f"VideoSaveWithAlpha: All PNG frames saved")

            # Step 2: Use FFmpeg to create video
            video_path = os.path.join(self.output_dir, output_filename)
            self._encode_video(frames_dir, video_path, frame_rate, format, quality)

            print(f"VideoSaveWithAlpha: Video saved to: {video_path}")

            # Step 3: Verify alpha channel (for formats that support it)
            if format in ["prores_ks", "libvpx-vp9"]:
                self._verify_alpha_channel(video_path, format)

            return (video_path,)

        finally:
            # Clean up temporary frames
            if frames_dir != folder_paths.get_temp_directory():
                try:
                    shutil.rmtree(frames_dir)
                    print(
                        f"VideoSaveWithAlpha: Cleaned up temp directory: {frames_dir}"
                    )
                except Exception as e:
                    print(
                        f"VideoSaveWithAlpha: Warning - failed to clean up temp directory: {e}"
                    )

    def _check_ram_disk(self):
        """Check if /dev/shm (RAM disk) is available"""
        return os.path.exists("/dev/shm") and os.access("/dev/shm", os.W_OK)

    def _get_extension(self, format):
        """Get file extension for format"""
        extensions = {
            "prores_ks": "mov",
            "libvpx-vp9": "webm",
            "qtrle": "mov",
            "png": "png",  # Will return list of PNG files
        }
        return extensions.get(format, "mp4")

    def _encode_video(self, frames_dir, video_path, frame_rate, format, quality):
        """Encode frames to video using FFmpeg"""

        # Construct FFmpeg command based on format
        if format == "prores_ks":
            # ProRes 4444 with alpha channel
            cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                str(frame_rate),
                "-i",
                os.path.join(frames_dir, "frame_%06d.png"),
                "-c:v",
                "prores_ks",
                "-profile:v",
                "4",  # ProRes 4444 (唯一支持alpha的profile)
                "-pix_fmt",
                "yuva444p10le",  # YUV 4:4:4 10-bit with Alpha
                "-qscale:v",
                str(quality),  # Quality (1-22)
                video_path,
            ]
            print(f"VideoSaveWithAlpha: Using ProRes 4444 with alpha channel")

        elif format == "libvpx-vp9":
            # VP9 with alpha channel (WebM format)
            cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                str(frame_rate),
                "-i",
                os.path.join(frames_dir, "frame_%06d.png"),
                "-c:v",
                "libvpx-vp9",
                "-pix_fmt",
                "yuva420p",  # YUV 4:2:0 with Alpha
                "-crf",
                str(quality),  # Quality (0-63, lower is better)
                "-b:v",
                "0",  # Constant quality mode
                video_path,
            ]
            print(f"VideoSaveWithAlpha: Using VP9 with alpha channel")

        elif format == "qtrle":
            # QuickTime Animation (RLE) with alpha
            cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                str(frame_rate),
                "-i",
                os.path.join(frames_dir, "frame_%06d.png"),
                "-c:v",
                "qtrle",
                video_path,
            ]
            print(f"VideoSaveWithAlpha: Using QuickTime Animation (RLE) with alpha")

        elif format == "png":
            # Just copy PNG files (no video encoding)
            print(f"VideoSaveWithAlpha: PNG sequence mode - no encoding needed")
            return

        print(f"VideoSaveWithAlpha: Running FFmpeg...")
        print(f"Command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            print(
                f"VideoSaveWithAlpha: FFmpeg failed with return code {result.returncode}"
            )
            print(f"STDERR: {result.stderr}")
            raise RuntimeError(
                f"FFmpeg failed to create video. "
                f"Check if ffmpeg is installed and supports the selected format."
            )

        print(f"VideoSaveWithAlpha: FFmpeg completed successfully")

    def _verify_alpha_channel(self, video_path, format):
        """Verify that the video has alpha channel"""

        try:
            # Use ffprobe to check pixel format
            cmd = [
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

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                pix_fmt = result.stdout.strip()
                print(f"VideoSaveWithAlpha: Output video pixel format: {pix_fmt}")

                # Check if format supports alpha
                if format == "prores_ks":
                    if pix_fmt.startswith("yuva444p"):
                        print(f"✓ Video has alpha channel ({pix_fmt})")
                    else:
                        print(f"⚠ Unexpected pixel format: {pix_fmt}")
                elif format == "libvpx-vp9":
                    if pix_fmt.startswith("yuva"):
                        print(f"✓ Video has alpha channel ({pix_fmt})")
                    else:
                        print(f"⚠ Unexpected pixel format: {pix_fmt}")
        except Exception as e:
            print(f"VideoSaveWithAlpha: Warning - could not verify alpha channel: {e}")


NODE_CLASS_MAPPINGS = {"VideoSaveWithAlpha": VideoSaveWithAlpha}

NODE_DISPLAY_NAME_MAPPINGS = {"VideoSaveWithAlpha": "Video Save (Alpha Support)"}
