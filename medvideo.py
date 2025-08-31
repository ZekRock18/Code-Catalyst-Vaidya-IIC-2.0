import streamlit as st
import requests
import json
import time
import os
import subprocess
import wave
import glob
import tempfile
import shutil
import re
import base64
from datetime import datetime
from typing import List, Dict
from google import genai
from google.genai import types
import yt_dlp

class MedicalVideoGenerator:
    def __init__(self, gemini_api_key: str):
        self.gemini_api_key = gemini_api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
        self.headers = {"Content-Type": "application/json"}
        self.request_delay = 6  # 6 seconds to respect 10 RPM limit
        
    def generate_medical_script(self, topic: str, language: str = "hindi") -> Dict:
        """Generate educational script and scene descriptions for medical topic"""
        
        if language == "english":
            lang_instruction = "in English"
            greeting = "Hello everyone! Today we'll learn about"
            closing = "Thank you for watching this educational content."
        else:
            lang_instruction = "in Hindi"
            greeting = "Namaskar dosto! Aaj hum seekhenge"
            closing = "Dhanyawad! Yeh tha aaj ka educational session."
        
        prompt = f"""Create a comprehensive educational script about "{topic}" for a medical educational video. The video should be informative, accurate, and suitable for students or general audiences learning about this medical topic.

Please provide:

1. NARRATOR SCRIPT ({lang_instruction}):
- Create a warm, educational 90-120 second narration {lang_instruction}
- Include proper medical terminology with simple explanations
- Make it engaging and easy to understand
- Start with a greeting like "{greeting}" and end with "{closing}"
- Include facts, processes, and key learning points
- Use appropriate language for the selected language option
- Do NOT include any call-to-action phrases like 'like', 'share', 'subscribe'
- Focus purely on educational content

2. 15 SCENE DESCRIPTIONS:
- Create 15 detailed scene descriptions for image generation
- Each scene should illustrate key aspects of "{topic}"
- Include medical diagrams, anatomical illustrations, or process visualization
- Make scenes educational and visually clear
- Each scene should be described for AI image generation
- If labels are needed, specify they should be clean, readable, and professional
- Describe medical illustrations with proper labeling when educational value requires it
- Ensure any text elements are described as clear and legible

Format your response as:

NARRATOR_SCRIPT:
[Your {language} script here]

SCENE_DESCRIPTIONS:
Scene 1: [Detailed description for image generation]
Scene 2: [Detailed description for image generation]
...
Scene 15: [Detailed description for image generation]

Make sure the content is medically accurate and educational."""

        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 4000}
        }
        
        try:
            response = requests.post(
                f"{self.base_url}?key={self.gemini_api_key}",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["candidates"][0]["content"]["parts"][0]["text"]
                
                # Parse the response
                parts = content.split("SCENE_DESCRIPTIONS:")
                if len(parts) == 2:
                    narrator_script = parts[0].replace("NARRATOR_SCRIPT:", "").strip()
                    scene_descriptions = parts[1].strip()
                    
                    return {
                        "success": True,
                        "narrator_script": narrator_script,
                        "scene_descriptions": scene_descriptions,
                        "full_content": content
                    }
                else:
                    return {"success": False, "error": "Could not parse script format"}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def create_scene_image_prompt(self, scene_description: str, scene_number: int) -> str:
        """Create image generation prompt for medical scene using enhanced model"""
        prompt = f"""Create an ultra-high quality, detailed medical educational illustration:

{scene_description}

CRITICAL Requirements:
- Ultra-realistic medical/educational illustration style with photographic quality
- Extremely detailed, crisp, and clear anatomical or process visualization
- Professional medical textbook quality imagery
- Vibrant, scientifically accurate colors suitable for learning
- Perfect 16:9 aspect ratio, 4K quality rendering
- Educational and scientifically precise
- Crystal clear visual elements
- Suitable for professional medical education video

TEXT AND LABELING REQUIREMENTS:
- If labels or text are needed, ensure they are:
  * Crystal clear, readable, and professional
  * Use clean, standard medical fonts
  * No glitched, corrupted, or distorted text
  * Proper spelling and medical terminology
  * Clean white or contrasting background for text readability
  * Straight, clean arrow lines pointing to anatomical parts
  * Text should be perfectly legible and anatomically accurate
- If including labels, make them look like professional medical textbook diagrams
- Ensure all text is sharp, clear, and properly formatted

VISUAL QUALITY:
- Use advanced rendering techniques for maximum detail and clarity
- Ensure every anatomical detail is perfectly accurate and visible
- Professional lighting and composition
- Medical illustration perfection with textbook-quality precision

Style: Ultra-high definition medical illustration, photorealistic quality, professional medical textbook standard with crystal-clear labeling if needed, maximum detail and clarity, perfect lighting and composition."""
        
        return prompt
    
    def generate_scene_image(self, prompt: str, scene_number: int) -> Dict:
        """Generate image using Gemini 2.0 Flash API"""
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"responseModalities": ["Text", "Image"]}
        }
        
        try:
            st.write(f"🎨 Generating image for scene {scene_number}...")
            response = requests.post(
                f"{self.base_url}?key={self.gemini_api_key}",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "scene_number": scene_number,
                    "response": result,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "scene_number": scene_number,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                "success": False,
                "scene_number": scene_number,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def save_image_from_response(self, response_data: Dict, output_dir: str) -> str:
        """Extract and save image from API response"""
        try:
            candidates = response_data.get("response", {}).get("candidates", [])
            if not candidates:
                raise ValueError("No candidates found in response")
            
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            
            for part in parts:
                if "inlineData" in part:
                    image_data = part["inlineData"]["data"]
                    mime_type = part["inlineData"]["mimeType"]
                    
                    image_bytes = base64.b64decode(image_data)
                    extension = "jpg" if "jpeg" in mime_type else mime_type.split("/")[-1]
                    filename = f"scene_{response_data['scene_number']:02d}.{extension}"
                    filepath = os.path.join(output_dir, filename)
                    
                    with open(filepath, "wb") as f:
                        f.write(image_bytes)
                    
                    return filepath
            
            raise ValueError("No image data found in response")
            
        except Exception as e:
            st.error(f"Error saving image for scene {response_data['scene_number']}: {str(e)}")
            return None
    
    def generate_audio(self, script: str, output_file: str, language: str = "hindi") -> bool:
        """Generate audio using Gemini TTS"""
        try:
            st.write("🎵 Generating audio narration...")
            
            client = genai.Client(api_key=self.gemini_api_key)
            
            # Set appropriate voice based on language
            if language == "english":
                voice_name = 'Kore'  # English voice
                tone_instruction = "Say in gentle, warm, and educational tone: "
            else:
                voice_name = 'Vindemiatrix'  # Hindi voice
                tone_instruction = "Say in gentle, warm, and educational tone: "
            
            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-tts",
                contents=f"{tone_instruction}{script}",
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice_name,
                            )
                        )
                    ),
                )
            )
            
            data = response.candidates[0].content.parts[0].inline_data.data
            
            # Save as WAV file
            with wave.open(output_file, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(data)
            
            st.success(f"Audio generated: {output_file}")
            return True
            
        except Exception as e:
            st.error(f"Error generating audio: {str(e)}")
            return False
    
    def get_audio_duration(self, audio_file):
        """Get the duration of a WAV audio file in seconds"""
        with wave.open(audio_file, 'rb') as wf:
            return wf.getnframes() / wf.getframerate()
    
    def natural_sort_key(self, filename):
        """Natural sorting key for filenames with numbers"""
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', filename)]
    
    def create_video_from_scenes(self, audio_file: str, images_folder: str, output_file: str) -> bool:
        """Create final video from scene images and audio"""
        try:
            st.write("🎬 Creating final video...")
            
            # Get all scene image files and sort them naturally
            image_pattern = os.path.join(images_folder, 'scene_*.png')
            image_files = sorted(glob.glob(image_pattern), key=self.natural_sort_key)
            
            if not image_files:
                image_pattern = os.path.join(images_folder, 'scene_*.jpg')
                image_files = sorted(glob.glob(image_pattern), key=self.natural_sort_key)
            
            if not image_files:
                st.error(f"No scene image files found in {images_folder}")
                return False
            
            st.write(f"Found {len(image_files)} scene images")
            
            # Get audio duration
            audio_duration = self.get_audio_duration(audio_file)
            st.write(f"Audio duration: {audio_duration:.2f} seconds")
            
            # Calculate duration for each image
            num_images = len(image_files)
            xfade_duration = 0.5
            
            if num_images > 1:
                image_duration = (audio_duration + ((num_images - 1) * xfade_duration)) / num_images
            else:
                image_duration = audio_duration
            
            st.write(f"Each scene duration: {image_duration:.2f} seconds")
            
            # Create FFmpeg filter complex string
            filter_complex = []
            inputs = []
            width, height, fps = 1920, 1080, 60
            
            # Add all images as inputs with scaling
            for i, img in enumerate(image_files):
                scale_cmd = f"[{i}:v]scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}[v{i}]"
                filter_complex.append(scale_cmd)
                inputs.extend(['-loop', '1', '-t', str(image_duration), '-i', img])
            
            # Create fade transitions between videos
            if num_images > 1:
                current_label = 'v0'
                
                for i in range(num_images - 1):
                    next_label = f'v{i+1}'
                    output_label = f'v{i+1}out' if i < num_images - 2 else 'vout'
                    
                    offset = i * (image_duration - xfade_duration) + (image_duration - xfade_duration)
                    
                    transition_cmd = (f"[{current_label}][{next_label}]xfade=transition=fade:"
                                    f"duration={xfade_duration}:offset={offset:.3f}[{output_label}]")
                    
                    filter_complex.append(transition_cmd)
                    current_label = output_label
                
                final_video_map = '[vout]'
            else:
                final_video_map = '[v0]'
            
            # Build the FFmpeg command
            cmd = [
                'ffmpeg', '-y',
                *inputs,
                '-i', audio_file,
                '-filter_complex', ';'.join(filter_complex),
                '-map', final_video_map,
                '-map', f'{num_images}:a',
                '-c:v', 'libx264',
                '-preset', 'slow',
                '-b:v', '10000k',
                '-maxrate', '12000k',
                '-bufsize', '20000k',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-pix_fmt', 'yuv420p',
                '-r', str(fps),
                '-aspect', '16:9',
                '-t', str(audio_duration),
                output_file
            ]
            
            # Execute FFmpeg command
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            st.success(f"Video created successfully: {output_file}")
            return True
            
        except subprocess.CalledProcessError as e:
            st.error(f"Error creating video: {e.stderr}")
            return False
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            return False
    
    def download_background_music(self, youtube_url: str, output_path: str = "bgaudio.mp3") -> bool:
        """Download background music from YouTube"""
        try:
            st.write("🎵 Downloading background music...")
            
            options = {
                'format': 'bestaudio/best',
                'outtmpl': output_path.replace('.mp3', '.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'quiet': True,
            }

            with yt_dlp.YoutubeDL(options) as ydl:
                ydl.download([youtube_url])
            
            st.success("Background music downloaded successfully")
            return True
            
        except Exception as e:
            st.error(f"Error downloading background music: {str(e)}")
            return False
    
    def merge_video_with_music(self, video_path: str, music_path: str, output_path: str) -> bool:
        """Merge video with background music"""
        try:
            st.write("🎼 Adding background music to video...")
            
            command = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-i', music_path,
                '-filter_complex',
                '[1:a]volume=0.1[a1];[0:a][a1]amix=inputs=2:duration=first[aout]',
                '-map', '0:v',
                '-map', '[aout]',
                '-c:v', 'libx264',
                '-b:v', '10000k',
                '-r', '60',
                '-vf', 'scale=1920:1080,setsar=1',
                '-preset', 'fast',
                '-c:a', 'aac',
                '-shortest',
                output_path
            ]

            subprocess.run(command, check=True, capture_output=True, text=True)
            st.success(f"Final video with music created: {output_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            st.error(f"Error merging video with music: {e}")
            return False
    
    def generate_complete_video(self, topic: str, language: str = "hindi") -> str:
        """Complete pipeline to generate educational medical video"""
        
        # Create working directories
        base_dir = "medical_video_output"
        scenes_dir = os.path.join(base_dir, "scenes")
        os.makedirs(scenes_dir, exist_ok=True)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Generate script and scene descriptions (10%)
            status_text.text("📝 Generating script and scene descriptions...")
            progress_bar.progress(0.1)
            
            script_result = self.generate_medical_script(topic, language)
            if not script_result["success"]:
                st.error(f"Failed to generate script: {script_result['error']}")
                return None
            
            narrator_script = script_result["narrator_script"]
            scene_descriptions = script_result["scene_descriptions"]
            
            st.success("Script generated successfully!")
            with st.expander("View Generated Script"):
                st.text(narrator_script)
            
            # Step 2: Generate audio (20%)
            status_text.text("🎵 Generating audio narration...")
            progress_bar.progress(0.2)
            
            audio_file = os.path.join(base_dir, "narration.wav")
            if not self.generate_audio(narrator_script, audio_file, language):
                st.error("Failed to generate audio")
                return None
            
            # Step 3: Download background music (25%)
            status_text.text("🎵 Downloading background music...")
            progress_bar.progress(0.25)
            
            music_file = os.path.join(base_dir, "bgaudio.mp3")
            background_music_url = "https://www.youtube.com/watch?v=zgEmUUmuh7Q"
            music_downloaded = self.download_background_music(background_music_url, music_file)
            
            # Step 4: Generate scene images (25% - 85%)
            status_text.text("🎨 Generating scene images...")
            
            # Parse scene descriptions
            scenes = []
            for line in scene_descriptions.split('\n'):
                if line.strip().startswith('Scene'):
                    scenes.append(line.strip())
            
            if len(scenes) < 15:
                st.warning(f"Only {len(scenes)} scenes found, expected 15. Proceeding with available scenes.")
            
            # Generate images for each scene
            successful_images = 0
            for i, scene_desc in enumerate(scenes[:15], 1):
                progress = 0.25 + (0.6 * i / 15)  # Progress from 25% to 85%
                progress_bar.progress(progress)
                status_text.text(f"🎨 Generating scene {i} of {len(scenes)}...")
                
                # Create image prompt
                image_prompt = self.create_scene_image_prompt(scene_desc, i)
                
                # Generate image
                result = self.generate_scene_image(image_prompt, i)
                
                if result["success"]:
                    # Save image
                    image_path = self.save_image_from_response(result, scenes_dir)
                    if image_path:
                        successful_images += 1
                        st.write(f"✅ Scene {i} generated")
                else:
                    st.warning(f"❌ Failed to generate scene {i}: {result.get('error', 'Unknown error')}")
                
                # Rate limiting delay
                if i < len(scenes):
                    time.sleep(self.request_delay)
            
            st.write(f"Successfully generated {successful_images}/{len(scenes)} scene images")
            
            # Step 5: Create video from scenes (90%)
            status_text.text("🎬 Creating video from scenes...")
            progress_bar.progress(0.9)
            
            video_file = os.path.join(base_dir, "educational_video.mp4")
            if not self.create_video_from_scenes(audio_file, scenes_dir, video_file):
                st.error("Failed to create video")
                return None
            
            # Step 6: Add background music (95%)
            final_video = video_file
            if music_downloaded and os.path.exists(music_file):
                status_text.text("🎼 Adding background music...")
                progress_bar.progress(0.95)
                
                final_video = os.path.join(base_dir, "final_video_with_music.mp4")
                if not self.merge_video_with_music(video_file, music_file, final_video):
                    st.warning("Failed to add background music, using video without music")
                    final_video = video_file
            
            # Step 7: Complete (100%)
            status_text.text("✅ Video generation complete!")
            progress_bar.progress(1.0)
            
            return final_video
            
        except Exception as e:
            st.error(f"Error in video generation pipeline: {str(e)}")
            return None

def main():
    st.set_page_config(
        page_title="Medical Educational Video Generator",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 Medical Educational Video Generator")
    st.markdown("Generate comprehensive educational videos on medical topics with AI-powered narration and visuals")
    
    # Use the API key 
    api_key = "YOUR_API_KEY"
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.success("✅ API Key configured")
        
        st.header("🎵 Optional Settings")
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            language = st.selectbox(
                "Narration Language", 
                ["hindi", "english"], 
                index=0,
                help="Select the language for script and audio narration"
            )
            video_quality = st.selectbox("Video Quality", ["1080p", "720p"], index=0)
            frame_rate = st.selectbox("Frame Rate", [60, 30], index=0)
    
    # Main interface
    st.header("📚 Generate Educational Video")
    
    # Topic input
    medical_topic = st.text_input(
        "Enter Medical Topic",
        placeholder="e.g., How the heart functions, Blood circulation system, Respiratory process...",
        help="Describe the medical topic you want to create an educational video about"
    )
    
    # Generate button
    if st.button("🚀 Generate Educational Video", type="primary"):
        if not medical_topic.strip():
            st.error("Please enter a medical topic")
            return
        
        # Initialize generator
        generator = MedicalVideoGenerator(api_key)
        
        # Create container for real-time updates
        with st.container():
            st.subheader(f"Generating video for: {medical_topic}")
            
            # Generate the complete video
            final_video_path = generator.generate_complete_video(
                topic=medical_topic,
                language=language
            )
            
            if final_video_path and os.path.exists(final_video_path):
                st.success("🎉 Video generation completed successfully!")
                
                # Show video details
                file_size = os.path.getsize(final_video_path) / (1024 * 1024)
                st.info(f"📊 Video Details:\n- File: {os.path.basename(final_video_path)}\n- Size: {file_size:.2f} MB\n- Language: {language.title()}")
                
                # Play video directly in the page
                st.subheader("🎬 Generated Educational Video")
                try:
                    # Read video file and display
                    with open(final_video_path, 'rb') as video_file:
                        video_bytes = video_file.read()
                    st.video(video_bytes)
                except Exception as e:
                    st.error(f"Error displaying video: {e}")
                    st.info(f"Video saved at: {final_video_path}")
                
                # Show generated files
                with st.expander("📁 View Generated Files"):
                    base_dir = "medical_video_output"
                    if os.path.exists(base_dir):
                        for root, dirs, files in os.walk(base_dir):
                            for file in files:
                                file_path = os.path.join(root, file)
                                rel_path = os.path.relpath(file_path, base_dir)
                                st.text(f"📄 {rel_path}")
            else:
                st.error("❌ Video generation failed. Please check the logs above.")
    
    # Information section
    st.markdown("---")
    st.subheader("ℹ️ How It Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🔄 Generation Process:**
        1. **Script Generation**: AI creates educational narration script in Hindi
        2. **Scene Planning**: Breaks content into 15 visual scenes
        3. **Audio Synthesis**: Generates professional narration
        4. **Image Creation**: Creates medical illustrations for each scene
        5. **Video Assembly**: Combines everything with smooth transitions
        6. **Music Integration**: Adds background music (optional)
        """)
    
    with col2:
        st.markdown("""
        **✨ Features:**
        - AI-powered medical content generation
        - Professional Hindi narration
        - High-quality medical illustrations
        - 1080p 60fps video output
        - Smooth scene transitions
        - Background music integration
        - Downloadable final video
        """)
    
    # Requirements section
    with st.expander("📋 Requirements & Setup"):
        st.markdown("""
        **Required Dependencies:**
        ```bash
        pip install streamlit requests google-generativeai yt-dlp
        ```
        
        **System Requirements:**
        - FFmpeg installed and accessible from command line
        - Google Gemini API key
        - Stable internet connection
        - Sufficient disk space for temporary files
        
        **API Setup:**
        - API key is pre-configured in the application
        - Ensure you have sufficient quota for image and audio generation
        - Rate limits: 10 requests per minute for image generation
        """)

if __name__ == "__main__":
    main()