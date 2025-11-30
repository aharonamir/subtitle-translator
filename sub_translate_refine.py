import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
# setting ollama host
os.environ['OLLAMA_IP']='192.168.1.15' # '172.26.186.61' # run  wsl hostname -I


import argparse
import pysrt
import tempfile
import subprocess
import re
from typing import List
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================

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
    
    # If no specific track_id provided, find first subtitle track
    if track_id is None:
        for line in result.stdout.split('\n'):
            if 'subtitles' in line:
                parts = line.split(':')[0].strip().split(' ')
                if len(parts) > 2:
                    track_id = parts[2]
                    print(f"Found subtitle track {track_id}")
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
                break
        if not track_found:
            raise ValueError(f"Track {track_id} not found or is not a subtitle track")
    
    # Extract subtitles using mkvextract
    subprocess.run([mkvextract_path, mkv_path, 'tracks', f'{track_id}:{output_path}'])

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

# ============================================================================
# TRANSLATION FUNCTIONS
# ============================================================================

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

# ============================================================================
# REFINEMENT FUNCTIONS
# ============================================================================

def refine_subtitles_with_ollama(
    original_srt_path: str,
    translated_srt_path: str,
    output_path: str,
    source_lang: str = "English",
    target_lang: str = "Hebrew",
    model: str = "qwen2.5:14b",
    batch_size: int = 15
):
    """
    Refine machine-translated subtitles with context awareness using Ollama.
    
    Args:
        original_srt_path: Path to original (English) SRT file
        translated_srt_path: Path to NLLB-translated SRT file
        output_path: Path to save refined subtitles
        source_lang: Source language name
        target_lang: Target language name
        model: Ollama model to use (recommended: qwen2.5:14b, llama3.3:70b)
        batch_size: Number of subtitles to process at once
    """
    try:
        from ollama import Client
        # import ollama
        ollama_ip_env = os.environ.get("OLLAMA_IP")
        ip = ollama_ip_env.strip().split()[0]
        host = f"http://{ip}:11434"
        print(f'Connecting to ollama at: {host}')
        ollama_client = Client(host=host)
    except ImportError:
        print("Error: Please install ollama package: pip install ollama")
        print("Also make sure Ollama is installed and running")
        return False
    
    print(f"Loading subtitles...")
    original_subs = pysrt.open(original_srt_path, encoding='utf-8')
    translated_subs = pysrt.open(translated_srt_path, encoding='utf-8')
    
    if len(original_subs) != len(translated_subs):
        print(f"Warning: Subtitle count mismatch! Original: {len(original_subs)}, Translated: {len(translated_subs)}")
    
    print(f"Refining {len(translated_subs)} subtitles with {model}...")
    refined_subs = pysrt.SubRipFile()
    
    for i in range(0, len(translated_subs), batch_size):
        batch_end = min(i + batch_size, len(translated_subs))
        batch_original = original_subs[i:batch_end]
        batch_translated = translated_subs[i:batch_end]
        
        # Include context from previous batch
        context_start = max(0, i - 3)
        context_subs = translated_subs[context_start:i] if i > 0 else []
        
        # Build prompt
        context_text = "\n".join([f"[{s.index}] {s.text}" for s in context_subs]) if context_subs else "(No previous context)"
        
        original_text = "\n".join([f"[{s.index}] {s.text}" for s in batch_original])
        translated_text = "\n".join([f"[{s.index}] {s.text}" for s in batch_translated])
        
        prompt = f"""You are refining machine-translated subtitles from {source_lang} to {target_lang}.
The machine translation may have issues with context, consistency, or naturalness.

Previous context:
{context_text}

Original {source_lang} subtitles:
{original_text}

Current {target_lang} translations (machine-generated):
{translated_text}

Please provide refined {target_lang} translations that:
1. Maintain consistency with previous context (names, terms, tone)
2. Sound natural and fluent in {target_lang}
3. Preserve the timing and meaning
4. Fix any grammatical or contextual errors

IMPORTANT: Return ONLY the refined translations in this exact format (no explanations):
[1] refined translation text
[2] refined translation text
[3] refined translation text

Start your response with the first subtitle index [{batch_original[0].index}]."""

        try:
            response = ollama_client.chat(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'temperature': 0.3,
                    'num_ctx': 4096,
                    'top_p': 0.9
                }
            )
            
            refined_batch = parse_llm_response(
                response['message']['content'],
                batch_translated
            )
            
            refined_subs.extend(refined_batch)
            print(f"  Processed {batch_end}/{len(translated_subs)} subtitles...")
            
        except Exception as e:
            print(f"  Error processing batch {i}-{batch_end}: {e}")
            print(f"  Using original translations for this batch")
            refined_subs.extend(batch_translated)
    
    # Save refined subtitles
    refined_subs.save(output_path, encoding='utf-8')
    print(f"\nRefinement complete! Saved to: {output_path}")
    return True

def parse_llm_response(response_text: str, original_batch: List) -> List:
    """Parse LLM response back into subtitle objects."""
    refined = []
    lines = response_text.strip().split('\n')
    
    # Pattern to match: [123] Translation text or just "123 Translation text"
    pattern = r'^\[?(\d+)\]?\s+(.+)$'
    
    current_index = None
    current_text_parts = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        match = re.match(pattern, line)
        if match:
            # If we have a previous subtitle being built, save it
            if current_index is not None and current_text_parts:
                text = '\n'.join(current_text_parts)
                orig_sub = next((s for s in original_batch if s.index == current_index), None)
                if orig_sub:
                    new_sub = pysrt.SubRipItem(
                        index=current_index,
                        start=orig_sub.start,
                        end=orig_sub.end,
                        text=text
                    )
                    refined.append(new_sub)
            
            # Start new subtitle
            current_index = int(match.group(1))
            current_text_parts = [match.group(2)]
        elif current_index is not None:
            # Continuation of previous subtitle (multi-line)
            current_text_parts.append(line)
    
    # Don't forget the last subtitle
    if current_index is not None and current_text_parts:
        text = '\n'.join(current_text_parts)
        orig_sub = next((s for s in original_batch if s.index == current_index), None)
        if orig_sub:
            new_sub = pysrt.SubRipItem(
                index=current_index,
                start=orig_sub.start,
                end=orig_sub.end,
                text=text
            )
            refined.append(new_sub)
    
    # Fallback: if parsing failed, return original batch
    if len(refined) != len(original_batch):
        print(f"    Warning: Parsed {len(refined)} subs but expected {len(original_batch)}")
        if len(refined) == 0:
            print(f"    Using original translations for this batch")
            return list(original_batch)
    
    return refined

def compare_subtitles(original_path: str, refined_path: str, num_samples: int = 5):
    """Show some examples of changes made during refinement."""
    original = pysrt.open(original_path, encoding='utf-8')
    refined = pysrt.open(refined_path, encoding='utf-8')
    
    print("\n" + "="*80)
    print("SAMPLE COMPARISONS (showing changes made)")
    print("="*80)
    
    changes_found = 0
    samples_shown = 0
    
    for orig, ref in zip(original, refined):
        if orig.text != ref.text and samples_shown < num_samples:
            changes_found += 1
            if samples_shown < num_samples:
                print(f"\nSubtitle #{orig.index}:")
                print(f"  BEFORE: {orig.text}")
                print(f"  AFTER:  {ref.text}")
                samples_shown += 1
    
    if changes_found == 0:
        print("\nNo changes were made (translations were already good!)")
    else:
        print(f"\n{changes_found} total subtitles were modified during refinement")
    print("="*80 + "\n")

# ============================================================================
# MUXING FUNCTIONS
# ============================================================================

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

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Translate MKV subtitles from English to Hebrew')
    parser.add_argument('mkv_file', help='Path to the MKV file')
    parser.add_argument('output_file', help='Path to save the translated SRT file')
    parser.add_argument('--mux', action='store_true', help='Mux translated subtitles back into the video file')
    parser.add_argument('--output-video', help='Output video file path (required if --mux is used)')
    parser.add_argument('--extract-audio', help='Extract audio track and convert to MP3 (provide output path)')
    parser.add_argument('--subtitle-track', type=int, help='Specific subtitle track ID to extract (default: first subtitle track)')
    parser.add_argument('--audio-track', type=int, help='Specific audio track ID to extract (default: first AAC track)')
    
    # Refinement options
    parser.add_argument('--refine', action='store_true', 
                       help='Refine translations with Ollama for better context awareness')
    parser.add_argument('--refine-model', default='qwen2.5:14b',
                       help='Ollama model for refinement (default: qwen2.5:14b)')
    parser.add_argument('--show-comparison', action='store_true',
                       help='Show before/after comparison of refined subtitles')
    
    args = parser.parse_args()
    
    if args.mux and not args.output_video:
        parser.error("--output-video is required when using --mux")
    
    # Create temporary file for extracted subtitles
    with tempfile.NamedTemporaryFile(suffix='.srt', delete=False) as temp_file:
        temp_path = temp_file.name
    
    # If refinement is enabled, we need an intermediate file
    intermediate_file = None
    if args.refine:
        intermediate_file = tempfile.NamedTemporaryFile(suffix='_nllb.srt', delete=False).name
    
    try:
        if args.output_file != "none":
            print("Extracting subtitles...")
            extract_subtitles(args.mkv_file, temp_path, args.subtitle_track)
            
            # Decide where to save NLLB output
            nllb_output = intermediate_file if args.refine else args.output_file
            
            print("Translating subtitles with NLLB...")
            process_subtitles(temp_path, nllb_output)
            
            # Refine with Ollama if requested
            if args.refine:
                print("\n" + "="*80)
                print("REFINING TRANSLATIONS WITH CONTEXT AWARENESS")
                print("="*80)
                print(f"Using model: {args.refine_model}")
                print("This will improve consistency, naturalness, and context understanding...\n")
                
                success = refine_subtitles_with_ollama(
                    original_srt_path=temp_path,
                    translated_srt_path=nllb_output,
                    output_path=args.output_file,
                    source_lang="English",
                    target_lang="Hebrew",
                    model=args.refine_model,
                    batch_size=15
                )
                
                if success and args.show_comparison:
                    compare_subtitles(nllb_output, args.output_file)
                
                # Clean up intermediate file
                if os.path.exists(nllb_output):
                    os.remove(nllb_output)
        else:
            args.output_file = None
            
        print(f"\nTranslation complete! Subtitles saved to {args.output_file}")
        
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
        # Clean up temporary files
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if intermediate_file and os.path.exists(intermediate_file):
            os.remove(intermediate_file)

if __name__ == '__main__':
    main()