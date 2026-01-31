# FFmpeg 8.0 下载和安装说明

## 为什么需要FFmpeg 8+

FFmpeg 8.0引入了完整的ProRes 4444 alpha支持。之前的版本中：
- ProRes 4444的alpha通道支持不完整
- 需要特定的像素格式才能正确处理透明背景

## 安装状态

✅ **FFmpeg 8.0已安装** - 预编译的Linux AMD64静态版本已下载并放置在`bin/ffmpeg`

### 验证安装

```bash
cd /var/tmp/vibe-kanban/worktrees/70e0-/Comfyui-Lyrics
./bin/ffmpeg -version | head -1
```

期望输出：
```
ffmpeg version n8.0.1-48-g0592d70bd
```

## 文件结构

```
Comfyui-Lyrics/
├── nodes.py
├── requirements.txt
├── README.md
├── MEMORY_OPTIMIZATION.md
├── FFMPEG_INSTALL.md    # 本文件
└── bin/
    └── ffmpeg          # FFmpeg 8.0二进制（188MB）
```

## 代码使用逻辑

`nodes.py`会自动检测并使用项目本地FFmpeg：
```python
# 优先使用bin/ffmpeg
if os.path.exists(local_ffmpeg):
    ffmpeg_binary = local_ffmpeg
else:
    ffmpeg_binary = "ffmpeg"  # 回退到系统FFmpeg
```

## 联系和支持

- FFmpeg官方文档: https://ffmpeg.org/documentation.html
- FFmpeg发布公告: https://ffmpeg.org/ffmpeg-8.0.html
- 项目issue tracker: https://github.com/ahkimkoo/Comfyui-Lyrics/issues

## 下载源（如需重新安装）

如需重新下载FFmpeg 8.0：

**官方GitHub发布**（推荐）:
```bash
cd /var/tmp/vibe-kanban/worktrees/70e0-/Comfyui-Lyrics
rm -rf bin ffmpeg* && mkdir -p bin
wget "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.0-latest-linux64-gpl-8.0.tar.xz"
xz -d ffmpeg-n8.0-latest-linux64-gpl-8.0.tar.xz
tar xf ffmpeg-n8.0-latest-linux64-gpl-8.0.tar
cp ffmpeg-n8.0-latest-linux64-gpl-8.0/bin/ffmpeg bin/
chmod +x bin/ffmpeg
```

**代理设置**（如需通过代理下载）:
```bash
export http_proxy="http://192.168.1.241:8118"
export https_proxy="http://192.168.1.241:8118"
# 然后运行上面的wget命令
```
