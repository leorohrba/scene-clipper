import os
import re
import subprocess
from faster_whisper import WhisperModel
import yt_dlp
import torch
from tqdm import tqdm
import time
from pyannote.audio import Pipeline
import warnings
from pyannote.audio.pipelines.utils.hook import ProgressHook

# from pyannote.audio.pipelines.utils import Pipeline

##############################
### CONFIGURATION SETTINGS ###
##############################

# Video configuration
VIDEO_URL = "https://www.youtube.com/watch?v=jKyMkheawlM"  # Replace with your video URL
OUTPUT_FILENAME = "subtitled_video.mp4"
SUBTITLE_STYLE = "FontSize=24,PrimaryColour=&HFFFFFF,Outline=1"

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Whisper model
WHISPER_MODEL_SIZE = "medium"  # Options: tiny, base, small, medium, large-v3
WHISPER_COMPUTE_TYPE = "int8"

# Speaker diarization
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
HF_API_TOKEN = ""  # Get from https://huggingface.co/settings/tokens

min_duration = 10.0  # Minimum duration for audio segments in seconds

##############################
### FUNCTION DEFINITIONS ###
##############################

def sanitize_filename(title):
    clean_title = re.sub(r'[^\w\-_\.]', '', title.replace(' ', '_'))
    return clean_title.strip()[:128]

def download_video():
    try:
        ydl_opts = {
            'quiet': True,
            'format': 'best[height<=720]',
            'outtmpl': 'downloaded_video.%(ext)s',
            'merge_output_format': 'mp4',
            'no_warnings': True,
            'ignoreerrors': True,
            'progress_hooks': [download_progress_hook],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(VIDEO_URL, download=True)
            downloaded_file = ydl.prepare_filename(info)
            
            for ext in ['.mp4', '.mkv', '.webm', '.mov']:
                candidate = f"downloaded_video{ext}"
                if os.path.exists(candidate):
                    return candidate

        raise Exception("No valid video file found after download")
    except Exception as e:
        print(f"❌ Download error: {str(e)}")
        return None

def download_progress_hook(d):
    if d['status'] == 'downloading':
        pbar = tqdm(
            total=d.get('total_bytes') or d.get('total_bytes_estimate'),
            unit='B', 
            unit_scale=True,
            desc="📥 Downloading video",
            leave=False
        )
        pbar.update(d['downloaded_bytes'] - pbar.n)
    elif d['status'] == 'finished':
        tqdm.write("✅ Download completed")

# Update the extract_audio function to handle short audio
def extract_audio(video_path):
    try:
        audio_path = os.path.splitext(video_path)[0] + "_audio.wav"
        
        # First pass: extract original audio
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            audio_path
        ]
        subprocess.run(cmd, check=True)

        # Check duration and pad if necessary
        duration = float(subprocess.check_output([
            'ffprobe', '-i', audio_path,
            '-show_entries', 'format=duration',
            '-v', 'quiet', '-of', 'csv=p=0'
        ]).decode().strip())

        # min_duration = 10.0  # Increased to 10 seconds
        if duration < min_duration:
            padded_path = audio_path.replace(".wav", "_padded.wav")
            pad_duration = min_duration - duration
        
            cmd = [
                'ffmpeg', '-y',
                '-i', audio_path,
                '-f', 'lavfi', '-t', str(pad_duration/2),
                '-i', 'anullsrc=r=16000:cl=mono',
                '-filter_complex',
                f'[0:a]afade=t=in:st=0:d=0.5[main];'
                f'[1:a]afade=t=in:st=0:d=0.5,afade=t=out:st={pad_duration/2-0.5}:d=0.5[pad];'
                f'[main][pad]concat=n=2:v=0:a=1',
                '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1',
                padded_path
            ]
            subprocess.run(cmd, check=True)
            audio_path = padded_path

        return audio_path
    except Exception as e:
        print(f"❌ Audio extraction failed: {str(e)}")
        return None

# Modify the diarization configuration
def perform_diarization(audio_path):
    try:
        # Check audio duration
        duration = float(subprocess.check_output([
            'ffprobe', '-i', audio_path,
            '-show_entries', 'format=duration',
            '-v', 'quiet', '-of', 'csv=p=0'
        ]).decode().strip())
        
        # For very short clips, skip diarization
        if duration < 5.0:
            print("⚠️ Audio too short for reliable diarization, skipping speaker identification")
            return None
         

        pipeline = Pipeline.from_pretrained(
            DIARIZATION_MODEL,
            use_auth_token=HF_API_TOKEN
        )

        pipeline.instantiate({
            'segmentation': {
                'min_duration_off': 0.0  # Allow shorter non-speech segments
            },
            'clustering': {
                'method': 'centroid',      
                'min_cluster_size': 10,    # Slightly reduced from default 12
                'threshold': 0.7           # Similar to default threshold
            }
        })
        with ProgressHook() as hook:
            # diarization = pipeline("audio.wav", hook=hook)
            return pipeline(audio_path, num_speakers=2, hook=hook)
    except Exception as e:
        print(f"❌ Diarization error: {str(e)}")
        return None

def generate_subtitles(video_path):
    try:
        print("🔊 Generating subtitles with speaker identification...")
        
        # Extract audio for diarization
        audio_path = extract_audio(video_path)
        diarization = perform_diarization(audio_path) if audio_path else None

        # Process diarization results
        diarization_segments = []
        speaker_mapping = {}
        current_label = 'A'
        
        if diarization:
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                diarization_segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker
                })
                if speaker not in speaker_mapping:
                    speaker_mapping[speaker] = f"Speaker {current_label}"
                    current_label = chr(ord(current_label) + 1)

        # Initialize Whisper model
        device = "cpu"
        model = WhisperModel(WHISPER_MODEL_SIZE,
                           device=device,
                           compute_type=WHISPER_COMPUTE_TYPE)

        subtitle_path = os.path.abspath('subtitles.srt')
        has_content = False

        def format_time(seconds):
            return f"{seconds//3600:02.0f}:{(seconds%3600)//60:02.0f}:{seconds%60:06.3f}".replace('.', ',')

        with open(subtitle_path, "w", encoding='utf-8') as f:
            counter = 1
            
            # Transcribe with progress
            with tqdm(desc="🔊 Transcribing audio", unit="seg") as pbar:
                segments, _ = model.transcribe(video_path,
                                             beam_size=5,
                                             word_timestamps=True)
                
                # Convert generator to list to get count
                segment_list = list(segments)
                pbar.total = len(segment_list)
                
                # Process segments
                for seg in segment_list:
                    if not seg.words:
                        continue

                    # Get precise timings
                    start_time = seg.words[0].start
                    end_time = seg.words[-1].end

                    # Determine speaker
                    speaker_label = ""
                    if diarization_segments:
                        speaker_durations = {}
                        for d in diarization_segments:
                            overlap_start = max(start_time, d['start'])
                            overlap_end = min(end_time, d['end'])
                            if overlap_start < overlap_end:
                                duration = overlap_end - overlap_start
                                speaker_durations[d['speaker']] = speaker_durations.get(d['speaker'], 0) + duration
                        
                        if speaker_durations:
                            dominant_speaker = max(speaker_durations, key=speaker_durations.get)
                            speaker_label = f"{speaker_mapping.get(dominant_speaker, 'Unknown')}: "

                    # Format subtitle text
                    subtitle_text = seg.text.strip()

                    # Write to file
                    f.write(f"{counter}\n")
                    f.write(f"{format_time(start_time)} --> {format_time(end_time)}\n")
                    f.write(f"{subtitle_text}\n\n")
                    
                    counter += 1
                    pbar.update(1)
                    has_content = True

        if not has_content:
            print("⚠️ No subtitles generated")
            os.remove(subtitle_path) if os.path.exists(subtitle_path) else None
            return None

        print(f"📝 Subtitles saved to: {subtitle_path}")
        return subtitle_path

    except Exception as e:
        print(f"❌ Subtitle error: {str(e)}")
        return None
def burn_subtitles(video_path, subtitle_path):
    try:
        print("🔥 Burning in subtitles...")
        
        abs_sub_path = os.path.abspath(subtitle_path)
        abs_video_path = os.path.abspath(video_path)
        
        output_file = OUTPUT_FILENAME
        
        # Get video duration for progress
        cmd = ['ffprobe', '-i', abs_video_path, '-show_entries', 'format=duration',
              '-v', 'quiet', '-of', 'csv=p=0']
        duration = float(subprocess.check_output(cmd).decode().strip())
        
        with tqdm(total=duration, desc="🔥 Burning subtitles", unit="sec") as pbar:
            process = subprocess.Popen(
                [
                    'ffmpeg',
                    '-hide_banner',
                    '-i', abs_video_path,
                    '-vf', f"subtitles=filename='{abs_sub_path}':force_style='{SUBTITLE_STYLE}'",
                    '-c:a', 'copy',
                    '-y',
                    output_file
                ],
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            while True:
                line = process.stderr.readline()
                if not line:
                    break
                time_match = re.search(r'time=(\d+:\d+:\d+\.\d+)', line)
                if time_match:
                    time_str = time_match.group(1)
                    current = sum(float(x) * 60 ** i for i, x in enumerate(reversed(time_str.split(":"))))
                    pbar.n = current
                    pbar.refresh()
            
            process.wait()
            return output_file
        
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg failed with error:\n{e.stderr}")
        return None
    except Exception as e:
        print(f"❌ Burn-in error: {str(e)}")
        return None

def main():
    print("⚙️ Initializing...")
    
    # Check dependencies
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        print("📦 Installing required packages...")
        subprocess.run(["pip", "install", "pyannote.audio"], check=True)
    
    # Download video
    video_path = download_video()
    if not video_path:
        return

    # Generate subtitles
    subtitle_path = generate_subtitles(video_path)
    if not subtitle_path:
        return

    # Burn subtitles into video
    final_output = burn_subtitles(video_path, subtitle_path)
    if final_output:
        print(f"✅ Success! Subtitled video saved as: {os.path.abspath(final_output)}")

if __name__ == "__main__":
    main()