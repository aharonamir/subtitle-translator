import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import argparse
import pysrt
import tempfile
import subprocess
import re
import ass
# from transformers import AutoTokenizer, MarianMTModel, AutoModelForSeq2SeqLM
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MarianMTModel  # For NLLB and OPUS
import torch

def convert_ass_to_srt(ass_path, srt_path):
    """Convert ASS/SSA subtitle file to SRT format.
    
    Args:
        ass_path: Path to ASS/SSA subtitle file
        srt_path: Path to save as SRT
    """
    def timestamp_to_srt(ass_timestamp):
        """Convert ASS timestamp (H:MM:SS.CC) to SRT timestamp (HH:MM:SS,mmm)"""
        parts = ass_timestamp.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds_and_centiseconds = parts[2].split('.')
        seconds = int(seconds_and_centiseconds[0])
        centiseconds = int(seconds_and_centiseconds[1]) if len(seconds_and_centiseconds) > 1 else 0
        milliseconds = centiseconds * 10
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    
    def strip_ass_formatting(text):
        """Remove ASS formatting tags from text"""
        text = re.sub(r'\{[^}]*\}', '', text)  # Remove all {...} tags
        text = text.replace('\\N', '\n').replace('\\n', '\n')  # Replace ASS newlines
        text = text.replace('\\h', ' ')  # Replace hard space with regular space
        return text.strip()
    
    try:
        with open(ass_path, 'r', encoding='utf-8-sig') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(ass_path, 'r', encoding='latin-1') as f:
            content = f.read()
    
    # Find the [Events] section
    events_match = re.search(r'\[Events\](.*?)(?:\[|$)', content, re.DOTALL)
    if not events_match:
        raise ValueError("No [Events] section found in ASS file")
    
    events_section = events_match.group(1)
    lines = events_section.split('\n')
    
    # Parse the format line to find column indices
    format_line = None
    for line in lines:
        if line.startswith('Format:'):
            format_line = line
            break
    
    if not format_line:
        raise ValueError("No Format line found in [Events] section")
    
    # Extract column names
    format_parts = [part.strip() for part in format_line.split(':', 1)[1].split(',')]
    
    start_idx = format_parts.index('Start') if 'Start' in format_parts else None
    end_idx = format_parts.index('End') if 'End' in format_parts else None
    text_idx = format_parts.index('Text') if 'Text' in format_parts else None
    
    if start_idx is None or end_idx is None or text_idx is None:
        raise ValueError("Could not find Start, End, or Text columns in Format line")
    
    # Parse subtitle events
    subtitles = []
    index = 1
    for line in lines:
        if line.startswith('Dialogue:') or line.startswith('Comment:'):
            parts = line.split(':', 1)[1].split(',', max(start_idx, end_idx, text_idx) + 1)
            
            if len(parts) > max(start_idx, end_idx, text_idx):
                start_time = parts[start_idx].strip()
                end_time = parts[end_idx].strip()
                text = ','.join(parts[text_idx + 1:]) if text_idx + 1 < len(parts) else parts[text_idx].strip()
                
                text = strip_ass_formatting(text)
                
                if text:  # Only add non-empty subtitles
                    subtitles.append({
                        'index': index,
                        'start': timestamp_to_srt(start_time),
                        'end': timestamp_to_srt(end_time),
                        'text': text
                    })
                    index += 1
    
    # Write SRT file
    with open(srt_path, 'w', encoding='utf-8') as f:
        for sub in subtitles:
            f.write(f"{sub['index']}\n")
            f.write(f"{sub['start']} --> {sub['end']}\n")
            f.write(f"{sub['text']}\n")
            f.write("\n")
    
    print(f"Converted ASS to SRT: {len(subtitles)} subtitles extracted")

def extract_subtitles(mkv_path, output_path, track_id=None):
    """Extract subtitles from MKV file.
    
    Args:
        mkv_path: Path to MKV file
        output_path: Path to save extracted subtitles
        track_id: Optional specific track ID to extract. If None, uses first subtitle track.
    """
    mkvextract_path = r"C:\Program Files\MKVToolNix\mkvextract.exe"
    if not os.path.exists(mkvextract_path):
        raise FileNotFoundError(f"mkvextract not found at {mkvextract_path}")

    # Get track info using mkvmerge
    mkvmerge_path = r"C:\Program Files\MKVToolNix\mkvmerge.exe"
    result = subprocess.run([mkvmerge_path, '-i', mkv_path], capture_output=True, text=True)
    print(result.stdout)
    
    # Track the format type
    format_type = None
    
    # If no specific track_id provided, find first subtitle track
    if track_id is None:
        for line in result.stdout.split('\n'):
            if 'subtitles' in line:
                parts = line.split(':')[0].strip().split(' ')
                if len(parts) > 2:
                    track_id = parts[2]
                    print(f"Found subtitle track {track_id}")
                    # Detect format type
                    if 'SubStationAlpha' in line or 'S_TEXT/ASS' in line:
                        format_type = 'ass'
                    elif 'S_TEXT/UTF8' in line or 'UTF-8' in line:
                        format_type = 'srt'
                    break
        
        if track_id is None:
            raise ValueError("No subtitle tracks found in the MKV file")
    else:
        # Verify the specified track exists and is a subtitle track
        track_found = False
        for line in result.stdout.split('\n'):
            if f'Track ID {track_id}' in line and 'subtitles' in line:
                track_found = True
                print(f"Using subtitle track {track_id}")
                if 'SubStationAlpha' in line or 'S_TEXT/ASS' in line:
                    format_type = 'ass'
                elif 'S_TEXT/UTF8' in line or 'UTF-8' in line:
                    format_type = 'srt'
                break
        if not track_found:
            raise ValueError(f"Track {track_id} not found or is not a subtitle track")
    
    # Extract subtitles using mkvextract
    # Use temporary file if ASS format to avoid encoding issues
    if format_type == 'ass':
        temp_extracted = tempfile.NamedTemporaryFile(suffix='.ass', delete=False).name
        subprocess.run([mkvextract_path, mkv_path, 'tracks', f'{track_id}:{temp_extracted}'])
        # Convert ASS to SRT
        convert_ass_to_srt(temp_extracted, output_path)
        # Clean up temp ASS file
        if os.path.exists(temp_extracted):
            os.remove(temp_extracted)
    else:
        subprocess.run([mkvextract_path, mkv_path, 'tracks', f'{track_id}:{output_path}'])

def load_translation_model_opus():
    """Load the translation model and tokenizer."""
    model_name = "Helsinki-NLP/opus-mt-en-he"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    return model, tokenizer, device

def translate_subtitle_opus(model, tokenizer, device, text):
    """Translate text from English to Hebrew using Hugging Face model."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512)
    
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation

def load_translation_model():
    """Load the translation model and tokenizer."""
    model_name = "facebook/nllb-200-distilled-600M"  # or "facebook/nllb-200-3.3B" for better quality
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    return model, tokenizer, device

def translate_subtitle(model, tokenizer, device, text):
    """Translate text from English to Hebrew using NLLB model."""
    # Add source and target language tokens
    src_lang = "eng_Latn"
    tgt_lang = "heb_Hebr"
    
    # Format input with language tokens
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
            max_length=512
        )
    
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation


def process_subtitles(input_path, output_path):
    """Process and translate subtitles."""
    print("Loading translation model...")
    model, tokenizer, device = load_translation_model()
    
    # Read original subtitles
    subs = pysrt.open(input_path)
    
    print("Translating subtitles...")
    # Translate each subtitle
    total = len(subs)
    for i, sub in enumerate(subs, 1):
        sub.text = translate_subtitle(model, tokenizer, device, sub.text)
        if i % 10 == 0:
            print(f"Translated {i}/{total} subtitles...")
    
    # Save translated subtitles
    subs.save(output_path, encoding='utf-8')

def mux_media(video_file, output_file, subtitle_file=None, audio_file=None):
    """Mux subtitles and/or audio into the video file."""
    mkvmerge_path = r"C:\Program Files\MKVToolNix\mkvmerge.exe"
    
    # Start with basic command
    command = [
        mkvmerge_path,
        '-o', output_file,  # Output file
        video_file,        # Original video
    ]
    
    # Add subtitle track if provided
    if subtitle_file:
        command.extend([
            '--language', '0:heb',  # Set subtitle language to Hebrew
            '--track-name', '0:Hebrew',  # Set track name
            '(',
            subtitle_file,
            ')'
        ])
    
    # Add audio track if provided
    if audio_file:
        command.extend([
            '--language', '0:heb',  # Set audio language to Hebrew
            '--track-name', '0:Hebrew Audio',  # Set track name
            '(',
            audio_file,
            ')'
        ])
    
    print("Muxing media files...")
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Warning: Muxing may have had some issues:")
        print(result.stderr)
    else:
        print("Muxing completed successfully!")

def extract_and_convert_audio(mkv_path, output_path, track_id=None):
    """Extract audio track and convert to MP3.
    
    Args:
        mkv_path: Path to MKV file
        output_path: Path to save converted MP3
        track_id: Optional specific track ID to extract. If None, uses first AAC track.
    """
    mkvmerge_path = r"C:\Program Files\MKVToolNix\mkvmerge.exe"
    result = subprocess.run([mkvmerge_path, '-i', mkv_path], capture_output=True, text=True)
    
    # Find first AAC audio track if no track_id provided
    if track_id is None:
        for line in result.stdout.split('\n'):
            if 'audio (AAC' in line:
                parts = line.split(':')[0].strip().split(' ')
                if len(parts) > 2:
                    track_id = parts[2]
                    print(f"Found AAC audio track {track_id}")
                    break
        
        if track_id is None:
            print("No AAC audio track found")
            return False
    else:
        # Verify the specified track exists and is an audio track
        track_found = False
        for line in result.stdout.split('\n'):
            if f'Track ID {track_id}' in line and 'audio' in line:
                track_found = True
                print(f"Using audio track {track_id}")
                break
        if not track_found:
            print(f"Track {track_id} not found or is not an audio track")
            return False
    
    # Extract audio using mkvextract
    temp_aac = tempfile.NamedTemporaryFile(suffix='.aac', delete=False).name
    mkvextract_path = r"C:\Program Files\MKVToolNix\mkvextract.exe"
    subprocess.run([mkvextract_path, mkv_path, 'tracks', f'{track_id}:{temp_aac}'])
    
    try:
        # Convert to MP3 using ffmpeg
        ffmpeg_path = "ffmpeg"  # Assuming ffmpeg is in PATH
        subprocess.run([
            ffmpeg_path,
            '-i', temp_aac,
            '-codec:a', 'libmp3lame',
            '-q:a', '2',  # VBR quality setting
            output_path
        ])
        print(f"Audio converted and saved to: {output_path}")
        return True
    finally:
        # Clean up temporary AAC file
        if os.path.exists(temp_aac):
            os.remove(temp_aac)

def main():
    parser = argparse.ArgumentParser(description='Translate MKV subtitles from English to Hebrew')
    parser.add_argument('mkv_file', help='Path to the MKV file')
    parser.add_argument('output_file', help='Path to save the translated SRT file')
    parser.add_argument('--mux', action='store_true', help='Mux translated subtitles back into the video file')
    parser.add_argument('--output-video', help='Output video file path (required if --mux is used)')
    parser.add_argument('--extract-audio', help='Extract audio track and convert to MP3 (provide output path)')
    parser.add_argument('--subtitle-track', type=int, help='Specific subtitle track ID to extract (default: first subtitle track)')
    parser.add_argument('--audio-track', type=int, help='Specific audio track ID to extract (default: first AAC track)')
    
    args = parser.parse_args()
    
    if args.mux and not args.output_video:
        parser.error("--output-video is required when using --mux")
    
    # If the input is already an SRT file, skip extraction and translate directly.
    input_is_srt = str(args.mkv_file).lower().endswith('.srt')
    if input_is_srt:
        if args.mux:
            parser.error("--mux cannot be used when input is an SRT file")

        if args.output_file != "none":
            print("Translating subtitles from provided SRT file...")
            process_subtitles(args.mkv_file, args.output_file)
            print(f"Translation complete! Translated subtitles saved to {args.output_file}")
        else:
            args.output_file = None
        return
    
    # Create temporary file for extracted subtitles
    with tempfile.NamedTemporaryFile(suffix='.srt', delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        if args.output_file != "none":
            print("Extracting subtitles...")
            extract_subtitles(args.mkv_file, temp_path, args.subtitle_track)
        
            print("Translating subtitles...")
            process_subtitles(temp_path, args.output_file)
        else:
            args.output_file = None
        print(f"Translation complete! Translated subtitles saved to {args.output_file}")
        
        # Extract and convert audio if requested
        audio_path = None
        if args.extract_audio:
            print("\nExtracting and converting audio track...")
            if extract_and_convert_audio(args.mkv_file, args.extract_audio, args.audio_track):
                audio_path = args.extract_audio
            else:
                print("Failed to extract audio track")
        
        # If muxing is requested, add subtitles and/or audio to the video
        if args.mux:
            print("\nMuxing media files...")
            mux_media(
                video_file=args.mkv_file,
                output_file=args.output_video,
                subtitle_file=args.output_file,
                audio_file=audio_path
            )
            print(f"\nFinal video saved to: {args.output_video}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    main()
