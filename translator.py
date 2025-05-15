import os
import re
import subprocess
from faster_whisper import WhisperModel
import yt_dlp
import torch
from tqdm import tqdm
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import hashlib
from pyannote.core import Annotation

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

# Translation model
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-tc-big-en-pt"

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

def compute_file_hash(file_path):
    """Compute MD5 hash of file contents"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


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

def extract_audio(video_path):
    try:
        audio_path = os.path.splitext(video_path)[0] + "_audio.wav"
        
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            audio_path
        ]
        subprocess.run(cmd, check=True)

        duration = float(subprocess.check_output([
            'ffprobe', '-i', audio_path,
            '-show_entries', 'format=duration',
            '-v', 'quiet', '-of', 'csv=p=0'
        ]).decode().strip())

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

def perform_diarization(audio_path):
    try:
        # Check cache first
        audio_hash = compute_file_hash(audio_path)
        cache_dir = "diarization_cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create unique cache key based on audio content and diarization config
        cache_key = f"{audio_hash}_{DIARIZATION_MODEL.replace('/', '_')}_spk2"
        cache_file = os.path.join(cache_dir, f"{cache_key}.rttm")
        
        if os.path.exists(cache_file):
            print("♻️ Loading diarization from cache")
            with open(cache_file, 'r') as f:
                return Annotation.from_rttm(f)[0]
            
        # Proceed with diarization if no cache
        duration = float(subprocess.check_output([
            'ffprobe', '-i', audio_path,
            '-show_entries', 'format=duration',
            '-v', 'quiet', '-of', 'csv=p=0'
        ]).decode().strip())
        
        if duration < 5.0:
            print("⚠️ Audio too short for reliable diarization, skipping speaker identification")
            return None

        pipeline = Pipeline.from_pretrained(
            DIARIZATION_MODEL,
            use_auth_token=HF_API_TOKEN
        )

        pipeline.instantiate({
            'segmentation': {
                'min_duration_off': 0.0
            },
            'clustering': {
                'method': 'centroid',      
                'min_cluster_size': 10,
                'threshold': 0.7
            }
        })
        
        with ProgressHook() as hook:
            diarization = pipeline(audio_path, num_speakers=2, hook=hook)
            
            # Save to cache for future runs
            with open(cache_file, 'w') as f:
                diarization.write_rttm(f)
            
            return diarization
            
    except Exception as e:
        print(f"❌ Diarization error: {str(e)}")
        return None

def generate_subtitles(video_path):
    try:
        print("🔊 Generating subtitles with translation...")
        
        audio_path = extract_audio(video_path)
        diarization = perform_diarization(audio_path) if audio_path else None

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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = WhisperModel(WHISPER_MODEL_SIZE,
                           device=device,
                           compute_type=WHISPER_COMPUTE_TYPE)

        # Initialize Helsinki-NLP translator
        try:
            print("⏳ Loading translation model...")
            tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL)
            translation_model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL)
            if torch.cuda.is_available():
                translation_model = translation_model.to(device)
        except Exception as e:
            print(f"⚠️ Translation model loading failed: {str(e)}")
            translation_model = None

        subtitle_path = os.path.abspath('subtitles.srt')
        has_content = False

        def format_time(seconds):
            return f"{seconds//3600:02.0f}:{(seconds%3600)//60:02.0f}:{seconds%60:06.3f}".replace('.', ',')

        def translate_text(text):
            if not translation_model:
                return text
                
            try:
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                if torch.cuda.is_available():
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                generated_tokens = translation_model.generate(
                    **inputs,
                    max_length=400
                )
                return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            except Exception as e:
                print(f"⚠️ Translation error: {str(e)}")
                return f"[TRANSLATION ERROR] {text}"

        with open(subtitle_path, "w", encoding='utf-8') as f:
            counter = 1
            
            with tqdm(desc="🔊 Transcribing audio", unit="seg") as pbar:
                segments, _ = model.transcribe(audio_path,
                                             beam_size=5,
                                             word_timestamps=True)
                
                segment_list = list(segments)
                pbar.total = len(segment_list)
                
                for seg in segment_list:
                    if not seg.words:
                        continue

                    start_time = seg.words[0].start
                    end_time = seg.words[-1].end

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

                    # Get original English text
                    subtitle_text = seg.text.strip()

                    # Translate to Portuguese
                    translated_text = translate_text(subtitle_text)

                    # Write translated text to subtitles
                    f.write(f"{counter}\n")
                    f.write(f"{format_time(start_time)} --> {format_time(end_time)}\n")
                    f.write(f"{translated_text}\n\n")
                    
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


def burn_subtitles(video_path, subtitle_path):
    try:
        print("🔥 Burning in subtitles...")
        
        abs_sub_path = os.path.abspath(subtitle_path)
        abs_video_path = os.path.abspath(video_path)
        
        output_file = OUTPUT_FILENAME
        
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
        from transformers import pipeline
    except ImportError:
        print("📦 Installing required packages...")
        subprocess.run(["pip", "install", "pyannote.audio", "transformers"], check=True)
    
    video_path = download_video()
    if not video_path:
        return

    subtitle_path = generate_subtitles(video_path)
    if not subtitle_path:
        return

    final_output = burn_subtitles(video_path, subtitle_path)
    if final_output:
        print(f"✅ Success! Subtitled video saved as: {os.path.abspath(final_output)}")

if __name__ == "__main__":
    main()