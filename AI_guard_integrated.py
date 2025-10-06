"""
AI Guard - Integrated System with Face Recognition, LLM, TTS, and ASR
This system recognizes faces and initiates conversations with unknown persons.
Supports: Google Gemini, OpenAI, and Anthropic
"""

import cv2
import face_recognition
import numpy as np
import pickle
import os
from pathlib import Path
from datetime import datetime
import threading
import queue
import pyaudio
import wave
import warnings
from pydub import AudioSegment
from pydub.playback import play
import io
import time

# Suppress FP16 warning from Whisper
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Set FFmpeg path for pydub (Windows fix)
if os.name == 'nt':  # Windows
    # Try common FFmpeg locations
    ffmpeg_paths = [
        r'C:\ffmpeg\bin\ffmpeg.exe',
        r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
        r'C:\ProgramData\chocolatey\bin\ffmpeg.exe',
    ]
    for ffmpeg_path in ffmpeg_paths:
        if os.path.exists(ffmpeg_path):
            AudioSegment.converter = ffmpeg_path
            print(f"✓ Found FFmpeg at: {ffmpeg_path}")
            break

# ASR
import whisper

# TTS
from gtts import gTTS
import tempfile

# LLM - Support for multiple providers
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None


class AIGuardConversation:
    def __init__(self, api_key= "AIzaSyDl4meToV6a8z6oooixUAH6aiDE2XeCADo", llm_provider="gemini", whisper_model="base"):
        """
        Initialize the conversational AI Guard
        
        Args:
            api_key: API key for LLM (Gemini, OpenAI, or Anthropic)
            llm_provider: "gemini", "openai", or "anthropic"
            whisper_model: "tiny", "base", "small", "medium", "large"
        """
        self.llm_provider = llm_provider
        self.api_key = api_key
        
        # Initialize LLM based on provider
        if llm_provider == "gemini":
            if genai is None:
                raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            print("✓ Using Google Gemini")
            
        elif llm_provider == "openai":
            if OpenAI is None:
                raise ImportError("openai not installed. Run: pip install openai")
            try:
                self.client = OpenAI(api_key=api_key)
                self.model_name = "gpt-4o-mini"
                print("✓ Using OpenAI GPT-4")
            except TypeError:
                import os
                os.environ["OPENAI_API_KEY"] = api_key
                self.client = OpenAI()
                self.model_name = "gpt-4o-mini"
                print("✓ Using OpenAI GPT-4")
                
        elif llm_provider == "anthropic":
            if anthropic is None:
                raise ImportError("anthropic not installed. Run: pip install anthropic")
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model_name = "claude-3-5-sonnet-20241022"
            print("✓ Using Anthropic Claude")
        
        # Initialize Whisper for ASR
        print(f"Loading Whisper model: {whisper_model}...")
        self.whisper_model = whisper.load_model(whisper_model)
        print("✓ Whisper model loaded")
        
        # Conversation history
        self.conversation_history = []
        self.is_conversing = False
        self.audio_queue = queue.Queue()
        
        # Audio settings
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        
    def text_to_speech(self, text):
        """Convert text to speech and play it"""
        try:
            print(f"🔊 AI Guard: {text}")
            
            # Generate speech using gTTS
            tts = gTTS(text=text, lang='en', slow=False)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                temp_filename = fp.name
                tts.save(temp_filename)
            
            # Try multiple playback methods
            try:
                # Method 1: pydub (requires FFmpeg)
                audio = AudioSegment.from_mp3(temp_filename)
                play(audio)
            except Exception as e1:
                try:
                    # Method 2: playsound (fallback, no FFmpeg needed)
                    import playsound
                    playsound.playsound(temp_filename)
                except ImportError:
                    try:
                        # Method 3: Windows native (only on Windows)
                        import winsound
                        # Convert mp3 to wav for winsound
                        print("  (Using Windows native audio - converting to WAV)")
                        import subprocess
                        wav_file = temp_filename.replace('.mp3', '.wav')
                        # Try using Windows Media Player
                        os.system(f'start /min wmplayer "{temp_filename}"')
                        time.sleep(len(text) * 0.1)  # Rough estimate of duration
                    except Exception as e3:
                        print(f"  ⚠ Audio playback failed: {e1}")
                        print(f"  Please install FFmpeg for proper audio playback")
            
            # Clean up
            try:
                os.unlink(temp_filename)
            except:
                pass
            
        except Exception as e:
            print(f"TTS Error: {e}")
    
    def record_audio(self, duration=5):
        """Record audio from microphone"""
        try:
            print("🎤 Listening... (speak now)")
            
            p = pyaudio.PyAudio()
            
            stream = p.open(format=self.FORMAT,
                          channels=self.CHANNELS,
                          rate=self.RATE,
                          input=True,
                          frames_per_buffer=self.CHUNK)
            
            frames = []
            
            for i in range(0, int(self.RATE / self.CHUNK * duration)):
                data = stream.read(self.CHUNK)
                frames.append(data)
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as fp:
                temp_filename = fp.name
                wf = wave.open(temp_filename, 'wb')
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(p.get_sample_size(self.FORMAT))
                wf.setframerate(self.RATE)
                wf.writeframes(b''.join(frames))
                wf.close()
            
            return temp_filename
            
        except Exception as e:
            print(f"Audio recording error: {e}")
            return None
    
    def speech_to_text(self, audio_file):
        """Convert speech to text using Whisper"""
        try:
            print("🔄 Processing speech...")
            result = self.whisper_model.transcribe(audio_file)
            text = result["text"].strip()
            print(f"👤 Person: {text}")
            return text
        except Exception as e:
            print(f"ASR Error: {e}")
            return ""
        finally:
            # Clean up audio file
            if os.path.exists(audio_file):
                os.unlink(audio_file)
    
    def get_llm_response(self, user_input):
        """Get response from LLM based on provider"""
        try:
            if self.llm_provider == "gemini":
                return self._get_gemini_response(user_input)
            elif self.llm_provider == "openai":
                return self._get_openai_response(user_input)
            elif self.llm_provider == "anthropic":
                return self._get_anthropic_response(user_input)
        except Exception as e:
            print(f"LLM Error: {e}")
            return "I'm having trouble processing that. Could you repeat?"
    
    def _get_gemini_response(self, user_input):
        """Get response from Google Gemini"""
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Build conversation for Gemini
        if len(self.conversation_history) == 1:
            # First message - include system prompt
            system_prompt = """You are a friendly AI receptionist helping visitors at a building entrance.
Your role is to:
1. Warmly greet visitors and ask their name and purpose
2. Verify they are expected or authorized
3. Keep responses brief, professional, and welcoming (1-2 sentences)
4. If they are legitimate visitors, welcome them warmly
5. If unclear about authorization, politely ask them to wait while you verify"""
            
            prompt = f"{system_prompt}\n\nVisitor: {user_input}\nReceptionist:"
        else:
            # Build conversation context
            prompt = "Conversation:\n"
            for msg in self.conversation_history:
                if msg["role"] == "user":
                    prompt += f"Visitor: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    prompt += f"Receptionist: {msg['content']}\n"
        
        try:
            # Configure safety settings to be more permissive
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                }
            ]
            
            # Get response from Gemini
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=150,
                    temperature=0.7,
                ),
                safety_settings=safety_settings
            )
            
            # Check if response was blocked
            if not response.candidates or not response.candidates[0].content.parts:
                print(f"⚠ Gemini safety filter triggered. Using fallback response.")
                print(f"   Finish reason: {response.candidates[0].finish_reason if response.candidates else 'Unknown'}")
                
                # Fallback to a safe, generic response
                if len(self.conversation_history) == 1:
                    assistant_message = "Hello! Welcome. May I ask who you are and the purpose of your visit today?"
                else:
                    assistant_message = "Thank you for that information. Could you please provide more details?"
            else:
                assistant_message = response.text.strip()
            
        except Exception as e:
            print(f"⚠ Gemini API error: {e}")
            # Fallback response
            if len(self.conversation_history) == 1:
                assistant_message = "Hello! Welcome. May I ask who you are and the purpose of your visit today?"
            else:
                assistant_message = "I understand. Could you tell me more about that?"
        
        # Add assistant response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return assistant_message
    
    def _get_openai_response(self, user_input):
        """Get response from OpenAI"""
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.conversation_history,
            max_tokens=150,
            temperature=0.7
        )
        assistant_message = response.choices[0].message.content
        
        # Add assistant response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return assistant_message
    
    def _get_anthropic_response(self, user_input):
        """Get response from Anthropic Claude"""
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Format for Anthropic
        system_message = self.conversation_history[0]["content"] if self.conversation_history[0]["role"] == "system" else ""
        messages = [msg for msg in self.conversation_history if msg["role"] != "system"]
        
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=150,
            system=system_message,
            messages=messages
        )
        assistant_message = response.content[0].text
        
        # Add assistant response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return assistant_message
    
    def start_conversation(self):
        """Start conversation with unknown person"""
        self.is_conversing = True
        
        # Initialize conversation with system prompt
        if self.llm_provider in ["openai", "anthropic"]:
            self.conversation_history = [
                {
                    "role": "system",
                    "content": """You are an AI security guard. An unknown person has been detected. 
Your job is to:
1. Politely ask who they are and their purpose
2. Determine if they are authorized to be here
3. Keep responses brief and conversational (1-2 sentences)
4. Be professional but friendly
5. If they seem legitimate, welcome them
6. If suspicious, inform them you're alerting security"""
                }
            ]
        else:
            self.conversation_history = []
        
        # Initial greeting
        initial_message = self.get_llm_response("An unknown person has been detected. Greet them politely and ask who they are.")
        self.text_to_speech(initial_message)
        
        # Conversation loop
        max_turns = 5
        turn = 0
        
        while self.is_conversing and turn < max_turns:
            # Record user response
            audio_file = self.record_audio(duration=5)
            
            if audio_file:
                # Convert to text
                user_text = self.speech_to_text(audio_file)
                
                if user_text:
                    # Get LLM response
                    response = self.get_llm_response(user_text)
                    
                    # Speak response
                    self.text_to_speech(response)
                    
                    turn += 1
                    
                    # Check if conversation should end
                    if any(word in response.lower() for word in ["welcome", "goodbye", "security", "alert"]):
                        break
                else:
                    self.text_to_speech("I didn't catch that. Could you speak again?")
            
            time.sleep(0.5)
        
        self.is_conversing = False
        print("Conversation ended.")
        
        # Log conversation
        self.log_conversation()
    
    def log_conversation(self):
        """Log the conversation to a file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"conversation_log_{timestamp}.txt"
        
        with open(log_file, 'w') as f:
            f.write(f"Conversation Log - {datetime.now()}\n")
            f.write("=" * 50 + "\n\n")
            
            for msg in self.conversation_history:
                if msg["role"] != "system":
                    role = "AI Guard" if msg["role"] == "assistant" else "Person"
                    f.write(f"{role}: {msg['content']}\n\n")
        
        print(f"✓ Conversation logged to {log_file}")


class IntegratedAIGuard:
    def __init__(self, api_key, llm_provider="openai", known_faces_dir="known_faces", 
                 encodings_file="face_encodings.pkl", whisper_model="base"):
        """Initialize the integrated AI Guard system"""
        self.known_faces_dir = known_faces_dir
        self.encodings_file = encodings_file
        self.known_face_encodings = []
        self.known_face_names = []
        
        Path(known_faces_dir).mkdir(exist_ok=True)
        
        # Initialize conversation system
        self.conversation_system = AIGuardConversation(
            api_key=api_key,
            llm_provider=llm_provider,
            whisper_model=whisper_model
        )
        
        # Track unknown faces
        self.unknown_face_detected = False
        self.last_conversation_time = 0
        self.conversation_cooldown = 60  # seconds between conversations
        
    def load_encodings(self):
        """Load face encodings from file"""
        if os.path.exists(self.encodings_file):
            with open(self.encodings_file, 'rb') as f:
                data = pickle.load(f)
                self.known_face_encodings = data['encodings']
                self.known_face_names = data['names']
            print(f"✓ Loaded {len(self.known_face_names)} known face(s)")
            return True
        return False
    
    def register_face_from_webcam(self, name, num_samples=5):
        """Register a face by capturing images from webcam"""
        print(f"Capturing {num_samples} samples for {name}")
        print("Press SPACE to capture, ESC to cancel")
        
        video_capture = cv2.VideoCapture(0)
        samples_captured = 0
        temp_encodings = []
        
        while samples_captured < num_samples:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Samples: {samples_captured}/{num_samples}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press SPACE to capture", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Register Face', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                print("Registration cancelled")
                video_capture.release()
                cv2.destroyAllWindows()
                return False
            
            elif key == 32:  # SPACE
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_encodings = face_recognition.face_encodings(rgb_frame)
                
                if len(face_encodings) > 0:
                    temp_encodings.append(face_encodings[0])
                    samples_captured += 1
                    print(f"  Captured sample {samples_captured}/{num_samples}")
                    
                    img_path = Path(self.known_faces_dir) / f"{name}_{samples_captured}.jpg"
                    cv2.imwrite(str(img_path), frame)
                else:
                    print("  No face detected, try again")
        
        video_capture.release()
        cv2.destroyAllWindows()
        
        if temp_encodings:
            avg_encoding = np.mean(temp_encodings, axis=0)
            self.known_face_encodings.append(avg_encoding)
            self.known_face_names.append(name)
            self.save_encodings()
            print(f"✓ Successfully registered {name}")
            return True
        
        return False
    
    def save_encodings(self):
        """Save face encodings to file"""
        data = {
            'encodings': self.known_face_encodings,
            'names': self.known_face_names
        }
        with open(self.encodings_file, 'wb') as f:
            pickle.dump(data, f)
    
    def recognize_and_interact(self, video_source=0, confidence_threshold=0.6):
        """Main function: recognize faces and initiate conversations with unknowns"""
        if not self.known_face_encodings:
            print("No faces registered! Please register faces first.")
            return
        
        print("\n" + "="*50)
        print("AI GUARD ACTIVE - Monitoring for unknown persons")
        print("="*50)
        print("Press 'q' to quit, 's' to save screenshot")
        
        video_capture = cv2.VideoCapture(video_source)
        frame_count = 0
        face_locations = []
        face_names = []
        unknown_detected_frames = 0
        unknown_threshold = 10  # Frames before starting conversation
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every other frame
            if frame_count % 2 == 0:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
                face_names = []
                has_unknown = False
                
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(
                        self.known_face_encodings, 
                        face_encoding, 
                        tolerance=confidence_threshold
                    )
                    name = "Unknown"
                    
                    face_distances = face_recognition.face_distance(
                        self.known_face_encodings, 
                        face_encoding
                    )
                    
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = self.known_face_names[best_match_index]
                            confidence = 1 - face_distances[best_match_index]
                            name = f"{name} ({confidence:.2f})"
                    
                    if "Unknown" in name:
                        has_unknown = True
                    
                    face_names.append(name)
                
                # Track unknown faces
                if has_unknown:
                    unknown_detected_frames += 1
                else:
                    unknown_detected_frames = 0
            
            # Draw results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                color = (0, 255, 0) if "Unknown" not in name else (0, 0, 255)
                
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            
            # Status display
            status = "ALERT: Unknown Person" if unknown_detected_frames > 0 else "All Clear"
            status_color = (0, 0, 255) if unknown_detected_frames > 0 else (0, 255, 0)
            cv2.putText(frame, status, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            
            cv2.imshow('AI Guard - Integrated System', frame)
            
            # Start conversation if unknown person detected for threshold frames
            current_time = time.time()
            if (unknown_detected_frames >= unknown_threshold and 
                not self.conversation_system.is_conversing and
                current_time - self.last_conversation_time > self.conversation_cooldown):
                
                print("\n🚨 UNKNOWN PERSON DETECTED - Initiating conversation...")
                self.last_conversation_time = current_time
                unknown_detected_frames = 0
                
                # Pause video and start conversation in separate thread
                conversation_thread = threading.Thread(
                    target=self.conversation_system.start_conversation
                )
                conversation_thread.start()
                conversation_thread.join()
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"alert_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
        
        video_capture.release()
        cv2.destroyAllWindows()


def main():
    """Main function"""
    print("=" * 60)
    print("AI GUARD - INTEGRATED SECURITY SYSTEM")
    print("Face Recognition + LLM + TTS + ASR")
    print("=" * 60)
    
    # # Get API key
    # print("\nChoose LLM provider:")
    # print("1. OpenAI (GPT-4)")
    # print("2. Anthropic (Claude)")
    
    # choice = input("Enter choice (1 or 2): ").strip()
    
    # if choice == '1':
    #     llm_provider = "openai"
    #     api_key = input("Enter your OpenAI API key: ").strip()
    # elif choice == '2':
    #     llm_provider = "anthropic"
    #     api_key = input("Enter your Anthropic API key: ").strip()
    # else:
    #     print("Invalid choice")
    #     return
    
    # if not api_key:
    #     print("API key required!")
    #     return
    
    # # Choose Whisper model
    # print("\nChoose Whisper model size:")
    # print("1. tiny (fastest, less accurate)")
    # print("2. base (balanced)")
    # print("3. small (more accurate)")
    # print("4. medium (very accurate, slower)")
    
    # whisper_choice = input("Enter choice (1-4, default=2): ").strip() or "2"
    # whisper_models = {"1": "tiny", "2": "base", "3": "small", "4": "medium"}
    # whisper_model = whisper_models.get(whisper_choice, "base")
    
    # Initialize system
    guard = IntegratedAIGuard(
        api_key="AIzaSyDl4meToV6a8z6oooixUAH6aiDE2XeCADo",
        llm_provider="gemini",
        whisper_model="base"
    )
    
    # Load or register faces
    if not guard.load_encodings():
        print("\nNo existing face data found.")
        name = input("Enter your name to register: ").strip()
        if name:
            guard.register_face_from_webcam(name, num_samples=5)
    
    if guard.known_face_encodings:
        print(f"\n✓ Known faces: {', '.join(guard.known_face_names)}")
        print("\nStarting AI Guard system...")
        print("The system will initiate conversation when unknown faces are detected.")
        input("Press Enter to start...")
        
        guard.recognize_and_interact(video_source=0)
    else:
        print("No faces registered. Exiting.")


if __name__ == "__main__":
    main()