# Subtitle Translator

This application extracts English subtitles from MKV files, translates them to Hebrew using the Helsinki-NLP translation model, and saves them as SRT files.

## Prerequisites

- Python 3.7+
- MKVToolNix (mkvextract) installed on your system
- Internet connection (for first-time model download)

## Installation

1. Install MKVToolNix:
   - Windows: Download and install from https://mkvtoolnix.download/
   - Make sure `mkvextract` is available in your system PATH

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage (Extract and Translate)
```bash
python subtitle_translator.py input.mkv output.srt
```

### Extract, Translate, and Mux into Video
```bash
python subtitle_translator.py input.mkv output.srt --mux --output-video output.mkv
```

Arguments:
- `input.mkv`: Path to your input MKV file
- `output.srt`: Path where you want to save the translated subtitles
- `--mux`: (Optional) Mux the translated subtitles back into the video
- `--output-video`: Path for the new video file with embedded Hebrew subtitles (required if --mux is used)

## Notes

- The script will only work with MKV files that contain English subtitles
- On first run, it will download the translation model (about 300MB)
- The translation is performed locally on your machine
- Uses the Helsinki-NLP English to Hebrew translation model

## Translate an existing SRT file

If you already have an English `.srt` file and only want to translate it to Hebrew (no MKV extraction), pass the `.srt` path as the first argument. Do not use `--mux` with an `.srt` input.

PowerShell example:

```powershell
python .\subtitle_translator.py .\Gangs.Of.London.S03E05.1080p.WEBRip.10bit.DDP5.1.x265-HODL.srt translated.srt
```

If you don't want to save the output file, pass `none` as the `output.srt` argument and the script will skip saving.

Example translating without saving:

```powershell
python .\subtitle_translator.py .\Gangs.Of.London.S03E05.1080p.WEBRip.10bit.DDP5.1.x265-HODL.srt none
```
