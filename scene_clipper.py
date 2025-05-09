import os
import re
import subprocess
from IPython.display import HTML, display
from faster_whisper import WhisperModel
import yt_dlp
from google.colab import files

##############################
### CONFIGURATION SETTINGS ###
##############################

# Scene configuration
SCENE_QUERY = "Terminator I'll be back"
TARGET_PHRASE = "I'll be back."  # Full exact quote
CLIP_DURATION = 90
BUFFER_BEFORE_PHRASE = 45

# Video processing
OUTPUT_FILENAME = SCENE_QUERY+".mp4"
SUBTITLE_STYLE = "FontSize=24,PrimaryColour=&HFFFFFF,Outline=1"

# Whisper model
WHISPER_MODEL_SIZE = "large-v3"
WHISPER_COMPUTE_TYPE = "float16" if check_cuda() else "int8"

##############################
### FUNCTION DEFINITIONS ###
##############################

def check_cuda():
    try:
        subprocess.check_output(['nvidia-smi'])
        return True
    except:
        return False

def sanitize_filename(title):
    # Replace spaces with underscores and remove special characters
    clean_title = re.sub(r'[^\w\-_\.]', '', title.replace(' ', '_'))
    return clean_title.strip()[:128]

def format_time(seconds):
    return f"{seconds//3600:02.0f}:{(seconds%3600)//60:02.0f}:{seconds%60:06.3f}".replace('.', ',')

def download_video():
    try:
        print(f"🔍 Searching YouTube for: '{SCENE_QUERY}'")
        base_filename = sanitize_filename(SCENE_QUERY)
        
        ydl_opts = {
            'quiet': True,
            'default_search': 'ytsearch1',
            'format': 'best',
            'outtmpl': f'{base_filename}.%(ext)s',
            'merge_output_format': 'mp4',
            'writesubtitles': True,
            'subtitleslangs': ['en.*'],
            'subtitlesformat': 'srt',
            'no_warnings': True,
            'ignoreerrors': True,
            'retries': 3,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(SCENE_QUERY, download=True)
            downloaded_file = ydl.prepare_filename(info)
            
            # Verify actual file existence with valid extension
            valid_extensions = ['.mp4', '.mkv', '.webm', '.mov']
            if not os.path.exists(downloaded_file):
                # Search for any video file with the base name
                for ext in valid_extensions:
                    candidate = f"{base_filename}{ext}"
                    if os.path.exists(candidate):
                        downloaded_file = candidate
                        break
                else:
                    raise FileNotFoundError("No video file downloaded")

        # Verify file is not empty
        if os.path.getsize(downloaded_file) == 0:
            raise ValueError("Downloaded file is empty")

        # Find subtitles
        base_name = os.path.splitext(downloaded_file)[0]
        for ext in ['.en.srt', '.en.vtt']:
            subtitle_file = base_name + ext
            if os.path.exists(subtitle_file):
                return downloaded_file, subtitle_file

        return downloaded_file, None

    except Exception as e:
        print(f"❌ Download error: {str(e)}")
        return None, None

def find_famous_moment(video_path):
    """Locate phrase start using flexible matching with word timestamps"""
    try:
        print(f"🔎 Analyzing for: '{TARGET_PHRASE}'...")
        
        device = "cuda" if check_cuda() else "cpu"
        model = WhisperModel(WHISPER_MODEL_SIZE, 
                           device=device, 
                           compute_type=WHISPER_COMPUTE_TYPE)
        
        # Transcribe with word timestamps
        segments, _ = model.transcribe(video_path, 
                                      beam_size=5,
                                      word_timestamps=True)
        
        # Normalization improvements
        def clean_word(w):
            w = w.lower()
            # Handle common contractions
            w = w.replace("'m", " am").replace("'re", " are")
            return re.sub(r'[^\w]', '', w)
        
        target_sequence = [clean_word(w) for w in TARGET_PHRASE.split()]
        print(f"🔦 Searching for sequence: {target_sequence}")

        # Collect all words with timestamps
        all_words = []
        for segment in segments:
            all_words.extend([
                {"text": clean_word(w.word), "start": w.start, "end": w.end}
                for w in segment.words
            ])

        # Flexible sequence matching
        seq_length = len(target_sequence)
        match_positions = []
        
        for i in range(len(all_words) - seq_length + 1):
            window = [w["text"] for w in all_words[i:i+seq_length]]
            if window == target_sequence:
                match_positions.append(all_words[i]["start"])

        if not match_positions:
            print("⚠️ No exact matches found. Trying partial match...")
            # Try matching any consecutive subset
            joined_target = " ".join(target_sequence)
            full_transcript = " ".join(w["text"] for w in all_words)
            if joined_target in full_transcript:
                start_idx = full_transcript.index(joined_target)
                words_before = len(full_transcript[:start_idx].split())
                if words_before < len(all_words):
                    return all_words[words_before]["start"]
                
        if match_positions:
            # Get earliest match
            exact_start = min(match_positions)
            print(f"🎯 Match found at {exact_start:.2f}s")
            return exact_start

        raise ValueError("Phrase not found in transcription")
    except Exception as e:
        print(f"❌ Detection error: {str(e)}")
        return None

def extract_clip(video_path, phrase_start):
    if phrase_start is None:
        print("⚠️ No phrase start time - extracting middle section")
        # Get video duration
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 
               'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
        duration = float(subprocess.check_output(cmd).decode('utf-8').strip())
        phrase_start = duration / 2  # Default to middle

    try:
        # Calculate ideal start time
        start_time = max(0, phrase_start - BUFFER_BEFORE_PHRASE)
        
        # Get video duration
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 
               'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
        duration = float(subprocess.check_output(cmd).decode('utf-8').strip())
        
        # Adjust if near end
        if start_time + CLIP_DURATION > duration:
            start_time = max(0, duration - CLIP_DURATION)

        print(f"✂️ Cutting from {start_time:.1f}s (phrase starts at {phrase_start:.1f}s)")
        
        cmd = [
            'ffmpeg',
            '-ss', str(start_time),
            '-i', video_path,
            '-t', str(CLIP_DURATION),
            '-c', 'copy',
            '-avoid_negative_ts', '1',
            '-y',
            'clipped_video.mp4'
        ]
        
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return 'clipped_video.mp4', start_time
    except Exception as e:
        print(f"❌ Clip error: {str(e)}")
        return None, None

def generate_subtitles(video_path, clip_start=0):
    try:
        print("🔊 Generating sentence-level subtitles...")
        
        device = "cuda" if check_cuda() else "cpu"
        model = WhisperModel(WHISPER_MODEL_SIZE,
                           device=device,
                           compute_type=WHISPER_COMPUTE_TYPE)

        # Get segment-level timings (full sentences)
        segments, _ = model.transcribe(video_path,
                                      beam_size=5,
                                      word_timestamps=False)  # Get full sentence timing

        subtitle_path = os.path.abspath('accurate_subtitles.srt')
        has_content = False

        def format_time(seconds):
            return f"{seconds//3600:02.0f}:{(seconds%3600)//60:02.0f}:{seconds%60:06.3f}".replace('.', ',')

        with open(subtitle_path, "w", encoding='utf-8') as f:
            counter = 1
            for seg in segments:
                # Get timing relative to clip
                start = seg.start
                end = seg.end
                
                # Filter segments outside clip duration
                if start > CLIP_DURATION or end < 0:
                    continue
                
                # Clamp times to clip boundaries
                start = max(0, start)
                end = min(CLIP_DURATION, end)
                
                # Skip invalid durations
                if end - start < 0.1:
                    continue

                f.write(f"{counter}\n")
                f.write(f"{format_time(start)} --> {format_time(end)}\n")
                f.write(f"{seg.text.strip()}\n\n")
                counter += 1
                has_content = True

        if not has_content:
            print("⚠️ No valid subtitles generated")
            if os.path.exists(subtitle_path):
                os.remove(subtitle_path)
            return None

        print(f"📝 Subtitles saved to: {subtitle_path}")
        return subtitle_path

    except Exception as e:
        print(f"❌ Subtitle error: {str(e)}")
        return None

def burn_subtitles(video_path, subtitle_path):
    try:
        print("🔥 Burning in subtitles...")
        
        if not subtitle_path or not os.path.exists(subtitle_path):
            raise FileNotFoundError(f"Subtitle file not found: {subtitle_path}")
            
        # Use absolute path and proper escaping
        abs_sub_path = os.path.abspath(subtitle_path)
        abs_video_path = os.path.abspath(video_path)
        
        cmd = [
            'ffmpeg',
            '-hide_banner',
            '-i', abs_video_path,
            '-vf', f"subtitles=filename='{abs_sub_path}':force_style='{SUBTITLE_STYLE}'",
            '-c:a', 'copy',
            '-y',
            OUTPUT_FILENAME
        ]
        
        # Run with error capture
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return OUTPUT_FILENAME
        
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg failed with error:\n{e.stderr}")
        return None
    except Exception as e:
        print(f"❌ Burn-in error: {str(e)}")
        return None

def display_and_download(video_path):
    try:
        print("🎥 Downloading final preview:")
        files.download(video_path)
    except Exception as e:
        print(f"❌ Display error: {str(e)}")


def main():
    # Setup environment
    print("⚙️ Initializing...")
    !pip install -q yt-dlp faster-whisper
    !apt install -y ffmpeg
    
    # Clean previous files
    for pattern in ['*_clip.mp4', '*.srt', '*.vtt']:
        !rm -f {pattern}
    
    # Download video (new unique filename)
    video_file, subs_file = download_video()
    if not video_file:
        print("💀 Failed to download video")
        return
    
    # Generate unique output name
    output_base = sanitize_filename(SCENE_QUERY)
    clipped_file = f"{output_base}_clip.mp4"
    final_output = f"{output_base}_final.mp4"
    
    # Find phrase timing
    phrase_time = find_famous_moment(video_file)
    
    # Extract clip containing the phrase
    clipped_file, clip_start = extract_clip(video_file, phrase_time)
    if not clipped_file:
        print("💀 Failed to create clip")
        return
    
    # Generate subtitles USING THE CLIPPED VIDEO (critical fix)
    subs_file = generate_subtitles(clipped_file)  # Always create new subs for clip
    
    # Burn subtitles if available
    final_file = burn_subtitles(clipped_file, subs_file) if subs_file else clipped_file
    
    # Output result
    if final_file:
        display_and_download(final_file)
        print(f"✅ Success! Saved as {final_file}")
    else:
        print("💀 Final processing failed")

if __name__ == "__main__":
    main()
