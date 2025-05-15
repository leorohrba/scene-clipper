import os
import re
import subprocess
from faster_whisper import WhisperModel
import yt_dlp
import webbrowser
import torch
import requests
from bs4 import BeautifulSoup
import pandas as pd

##############################
### CONFIGURATION SETTINGS ###
##############################

# Scene configuration
SCENE_QUERY = "Karate Kid wax on wax off"  # Search query
TARGET_PHRASE = "Wax on, wax off."  # Full exact quote
CLIP_DURATION = 90
BUFFER_BEFORE_PHRASE = 45

# Video processing
OUTPUT_FILENAME = SCENE_QUERY+".mp4"
SUBTITLE_STYLE = "FontSize=24,PrimaryColour=&HFFFFFF,Outline=1"

# Whisper model - use smaller model sizes for less resource usage
WHISPER_MODEL_SIZE = "medium"  # Options: tiny, base, small, medium, large-v3
WHISPER_COMPUTE_TYPE = "int8"  # Less memory intensive

##############################
### FUNCTION DEFINITIONS ###
##############################
def fetch_famous_quotes(limit=10):
    """Fetch famous movie quotes from Wikipedia AFI list"""
    print("📚 Fetching famous movie quotes...")
    
    url = "https://en.wikipedia.org/wiki/AFI%27s_100_Years...100_Movie_Quotes"
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the quotes table
        tables = soup.find_all('table', {'class': 'wikitable'})
        if not tables:
            raise ValueError("Could not find quotes table on Wikipedia page")
        
        quotes_table = tables[0]
        
        # Extract table data
        quotes_data = []
        rows = quotes_table.find_all('tr')[1:]  # Skip header row
        
        for row in rows[:limit]:  # Limit to requested number of quotes
            cells = row.find_all(['th', 'td'])
            if len(cells) >= 6:
                rank = cells[0].text.strip()
                quote = cells[1].text.strip().replace('"', '')
                character = cells[2].text.strip()
                actor = cells[3].text.strip()
                film = cells[4].text.strip()
                year = cells[5].text.strip()
                
                quotes_data.append({
                    'rank': rank,
                    'quote': quote,
                    'character': character,
                    'actor': actor,
                    'film': film,
                    'year': year,
                    'search_query': f"{film} {quote}",
                })
        
        print(f"✅ Found {len(quotes_data)} movie quotes")
        return quotes_data
    
    except Exception as e:
        print(f"❌ Error fetching quotes: {str(e)}")
        return []

def check_cuda():
    try:
        return torch.cuda.is_available()
    except:
        return False

def sanitize_filename(title):
    clean_title = re.sub(r'[^\w\-_\.]', '', title.replace(' ', '_'))
    return clean_title.strip()[:128]

def download_video():
    try:
        print(f"🔍 Searching YouTube for: '{SCENE_QUERY}'")
        base_filename = sanitize_filename(SCENE_QUERY)
        
        ydl_opts = {
            'quiet': True,
            'default_search': 'ytsearch1',
            'format': 'best[height<=720]',  # Limit to 720p to save resources
            'outtmpl': f'{base_filename}.%(ext)s',
            'merge_output_format': 'mp4',
            # 'writesubtitles': True,
            # 'subtitleslangs': ['en.*'],
            # 'subtitlesformat': 'srt',
            'no_warnings': True,
            'ignoreerrors': True,
            'retries': 3,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(SCENE_QUERY, download=True)
            downloaded_file = ydl.prepare_filename(info)
            
            for ext in ['.mp4', '.mkv', '.webm', '.mov']:
                candidate = f"{base_filename}{ext}"
                if os.path.exists(candidate):
                    downloaded_file = candidate
                    break

        if os.path.getsize(downloaded_file) == 0:
            raise ValueError("Downloaded file is empty")

        # base_name = os.path.splitext(downloaded_file)[0]
        # for ext in ['.en.srt', '.en.vtt']:
        #     subtitle_file = base_name + ext
        #     if os.path.exists(subtitle_file):
        #         return downloaded_file, subtitle_file

        return downloaded_file, None

    except Exception as e:
        print(f"❌ Download error: {str(e)}")
        return None, None

def find_famous_moment(video_path):
    try:
        print(f"🔎 Analyzing for: '{TARGET_PHRASE}'...")
        
        device = "cuda" if check_cuda() else "cpu"
        print(f"Using device: {device} with model: {WHISPER_MODEL_SIZE}")
        
        model = WhisperModel(WHISPER_MODEL_SIZE, 
                           device=device, 
                           compute_type=WHISPER_COMPUTE_TYPE)
        
        segments, _ = model.transcribe(video_path, 
                                      beam_size=5,
                                      word_timestamps=True)
        
        def clean_word(w):
            w = w.lower()
            w = w.replace("'m", " am").replace("'re", " are")
            return re.sub(r'[^\w]', '', w)
        
        target_sequence = [clean_word(w) for w in TARGET_PHRASE.split()]
        print(f"🔦 Searching for sequence: {target_sequence}")

        all_words = []
        for segment in segments:
            all_words.extend([
                {"text": clean_word(w.word), "start": w.start, "end": w.end}
                for w in segment.words
            ])

        seq_length = len(target_sequence)
        match_positions = []
        
        for i in range(len(all_words) - seq_length + 1):
            window = [w["text"] for w in all_words[i:i+seq_length]]
            if window == target_sequence:
                match_positions.append(all_words[i]["start"])

        if not match_positions:
            joined_target = " ".join(target_sequence)
            full_transcript = " ".join(w["text"] for w in all_words)
            if joined_target in full_transcript:
                start_idx = full_transcript.index(joined_target)
                words_before = len(full_transcript[:start_idx].split())
                if words_before < len(all_words):
                    return all_words[words_before]["start"]
                
        if match_positions:
            exact_start = min(match_positions)
            print(f"🎯 Match found at {exact_start:.2f}s")
            return exact_start

        raise ValueError("Phrase not found in transcription")
    except Exception as e:
        print(f"❌ Detection error: {str(e)}")
        return None

def extract_clip(video_path, phrase_start):
    if phrase_start is None:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 
               'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
        duration = float(subprocess.check_output(cmd).decode('utf-8').strip())
        phrase_start = duration / 2

    try:
        start_time = max(0, phrase_start - BUFFER_BEFORE_PHRASE)
        
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 
               'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
        duration = float(subprocess.check_output(cmd).decode('utf-8').strip())
        
        if start_time + CLIP_DURATION > duration:
            start_time = max(0, duration - CLIP_DURATION)

        print(f"✂️ Cutting from {start_time:.1f}s (phrase starts at {phrase_start:.1f}s)")
        
        output_file = sanitize_filename(SCENE_QUERY) + "_clip.mp4"
        
        cmd = [
            'ffmpeg',
            '-ss', str(start_time),
            '-i', video_path,
            '-t', str(CLIP_DURATION),
            '-c', 'copy',
            '-avoid_negative_ts', '1',
            '-y',
            output_file
        ]
        
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output_file, start_time
    except Exception as e:
        print(f"❌ Clip error: {str(e)}")
        return None, None

def generate_subtitles(video_path, clip_start=0):
    try:
        print("🔊 Generating accurate subtitles...")
        
        device = "cuda" if check_cuda() else "cpu"
        model = WhisperModel(WHISPER_MODEL_SIZE,
                           device=device,
                           compute_type=WHISPER_COMPUTE_TYPE)

        # Get word-level timings for precision
        segments, _ = model.transcribe(video_path,
                                      beam_size=5,
                                      word_timestamps=True)

        subtitle_path = os.path.abspath('accurate_subtitles.srt')
        has_content = False

        def format_time(seconds):
            return f"{seconds//3600:02.0f}:{(seconds%3600)//60:02.0f}:{seconds%60:06.3f}".replace('.', ',')
        
        def correct_names(text):
            # Add more name corrections as needed
            corrections = {
                "Red": "Rhett",
                "red": "Rhett"
            }
            
            for wrong, right in corrections.items():
                # Use word boundaries to avoid partial replacements
                text = re.sub(r'\b' + wrong + r'\b', right, text)
            return text

        with open(subtitle_path, "w", encoding='utf-8') as f:
            counter = 1
            first_subtitle_processed = False
            
            for seg in segments:
                if not seg.words:
                    continue

                # Get precise timings from first word
                start_time = seg.words[0].start
                end_time = seg.words[-1].end

                # For first subtitle only, add slight delay
                if not first_subtitle_processed:
                    start_time = max(0.1, start_time)  # Add 100ms buffer
                    first_subtitle_processed = True

                start_time = max(0, start_time)
                end_time = min(CLIP_DURATION, end_time)

                if end_time - start_time < 0.1:
                    continue

                f.write(f"{counter}\n")
                f.write(f"{format_time(start_time)} --> {format_time(end_time)}\n")
                # f.write(f"{seg.text.strip()}\n\n")

                corrected_text = correct_names(seg.text.strip())
                f.write(f"{corrected_text}\n\n")
                
                counter += 1
                has_content = True

        if not has_content:
            print("⚠️ No valid subtitles generated")
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
        
        output_file = sanitize_filename(SCENE_QUERY) + "_final.mp4"
        
        cmd = [
            'ffmpeg',
            '-hide_banner',
            '-i', abs_video_path,
            '-vf', f"subtitles=filename='{abs_sub_path}':force_style='{SUBTITLE_STYLE}'",
            '-c:a', 'copy',
            '-y',
            output_file
        ]
        
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return output_file
        
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg failed with error:\n{e.stderr}")
        return None
    except Exception as e:
        print(f"❌ Burn-in error: {str(e)}")
        return None

def process_movie_quote(quote_data):
    """Process a single movie quote"""
    global SCENE_QUERY, TARGET_PHRASE
    
    # Update global variables for this quote
    SCENE_QUERY = f"{quote_data['film']} {quote_data['quote'][:40]}"
    TARGET_PHRASE = quote_data['quote']
    
    print(f"\n\n{'='*50}")
    print(f"📽️ Processing #{quote_data['rank']}: \"{quote_data['quote']}\"")
    print(f"🎬 From: {quote_data['film']} ({quote_data['year']}) - {quote_data['character']}")
    print(f"{'='*50}\n")
    
    # Standard processing pipeline
    video_file, subs_file = download_video()
    if not video_file:
        print("💀 Failed to download video")
        return False
    
    phrase_time = find_famous_moment(video_file)
    
    clipped_file, clip_start = extract_clip(video_file, phrase_time)
    if not clipped_file:
        print("💀 Failed to create clip")
        return False
    
    subs_file = generate_subtitles(clipped_file)
    
    output_file = sanitize_filename(f"{quote_data['rank']}_{quote_data['film']}_{quote_data['quote'][:20]}")
    final_file = burn_subtitles(clipped_file, subs_file) if subs_file else clipped_file
    
    if final_file:
        # Rename to include rank for better sorting
        if os.path.exists(final_file):
            # Convert rank to integer before formatting
            try:
                rank_num = int(quote_data['rank'])
                new_name = f"quote_{rank_num:03d}_{os.path.basename(final_file)}"
            except ValueError:
                # If rank cannot be converted to int, use it as-is
                new_name = f"quote_{quote_data['rank']}_{os.path.basename(final_file)}"
            
            os.rename(final_file, new_name)
            final_file = new_name
            
        print(f"✅ Success! File saved as: {os.path.abspath(final_file)}")
        return True
    else:
        print("💀 Final processing failed")
        return False

def main():
    print("⚙️ Initializing...")
    
    # Install dependencies if needed
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        print("📦 Installing required packages...")
        subprocess.run(["pip", "install", "requests", "BeautifulSoup", "pandas"], 
                      check=True, capture_output=True)
        
        import requests
        from bs4 import BeautifulSoup
    
    # Get quotes to process
    quotes = fetch_famous_quotes(limit=1)  # Start with just 2
    
    if not quotes:
        print("No quotes found to process")
        return
    
    # Process each quote
    success_count = 0
    for i, quote in enumerate(quotes):
        print(f"\nProcessing quote {i+1}/{len(quotes)}")
        if process_movie_quote(quote):
            success_count += 1
    
    print(f"\n✨ Processing complete! Successfully processed {success_count}/{len(quotes)} quotes.")


if __name__ == "__main__":
    main()