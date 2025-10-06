"""
AI Guard Agent - Main Entry Point
EE782 Programming Assignment 2

This is the main file that orchestrates the AI Guard system.
It follows the milestone structure from the assignment:
- Milestone 1: Activation via speech command
- Milestone 2: Face recognition and enrollment
- Milestone 3: Escalation dialogue with unknown persons

Usage:
    python main.py
"""

import sys
import os
from guard_functions import (
    GuardConfig,
    initialize_whisper,
    listen_for_command,
    load_face_encodings,
    save_face_encodings,
    enroll_face_from_webcam,
    monitor_room,
    speak
)


def print_banner():
    """Print welcome banner"""
    print("\n" + "="*70)
    print("    AI GUARD AGENT - INTELLIGENT ROOM MONITORING SYSTEM")
    print("    EE782: Advanced Topics in Machine Learning")
    print("="*70)
    print("\n    Components:")
    print("    - Face Recognition (Milestone 2)")
    print("    - Speech Recognition (Milestone 1)")
    print("    - Text-to-Speech")
    print("    - Conversational AI with Escalation (Milestone 3)")
    print("="*70 + "\n")


def setup_face_enrollment(config):
    """
    Milestone 2: Face Recognition and Enrollment
    
    Args:
        config: GuardConfig object
    
    Returns:
        tuple: (known_encodings, known_names)
    """
    print("\n" + "-"*70)
    print("MILESTONE 2: FACE RECOGNITION SETUP")
    print("-"*70)
    
    # Try to load existing encodings
    known_encodings, known_names = load_face_encodings(config)
    
    if known_encodings:
        print(f"\nFound {len(known_names)} enrolled face(s):")
        for i, name in enumerate(known_names, 1):
            print(f"  {i}. {name}")
        
        # Ask if user wants to add more
        while True:
            choice = input("\nAdd another trusted person? (y/n): ").strip().lower()
            if choice == 'y':
                name = input("Enter name: ").strip()
                if name:
                    encoding = enroll_face_from_webcam(config, name, num_samples=5)
                    if encoding is not None:
                        known_encodings.append(encoding)
                        known_names.append(name)
                        save_face_encodings(config, known_encodings, known_names)
            elif choice == 'n':
                break
            else:
                print("Please enter 'y' or 'n'")
    
    else:
        # No existing faces - must enroll at least one
        print("\nNo trusted faces found. You must enroll at least one person.")
        
        while not known_encodings:
            name = input("\nEnter name of trusted person: ").strip()
            if name:
                encoding = enroll_face_from_webcam(config, name, num_samples=5)
                if encoding is not None:
                    known_encodings.append(encoding)
                    known_names.append(name)
                    save_face_encodings(config, known_encodings, known_names)
                else:
                    print("Enrollment failed. Please try again.")
            else:
                print("Name cannot be empty.")
    
    print(f"\n✓ Face enrollment complete. {len(known_names)} trusted person(s) enrolled.")
    return known_encodings, known_names


def wait_for_activation(whisper_model, config, activation_keyword="guard my room"):
    """
    Milestone 1: Wait for activation command
    
    Args:
        whisper_model: Loaded WhisperModel instance
        config: GuardConfig object
        activation_keyword: Phrase to activate guard mode
    
    Returns:
        bool: True if activated successfully
    """
    print("\n" + "-"*70)
    print("MILESTONE 1: VOICE ACTIVATION")
    print("-"*70)
    print(f"\nActivation keyword: '{activation_keyword}'")
    print("The system will listen for this command to enter guard mode.\n")
    
    max_attempts = 3
    attempt = 0
    
    while attempt < max_attempts:
        attempt += 1
        print(f"Attempt {attempt}/{max_attempts}")
        
        if listen_for_command(whisper_model, config, activation_keyword):
            return True
        else:
            print(f"Activation keyword not detected.")
            if attempt < max_attempts:
                print("Please try again...\n")
    
    print(f"\nFailed to activate after {max_attempts} attempts.")
    return False


def main():
    """Main function - orchestrates the entire AI Guard system"""
    
    try:
        # Print welcome banner
        print_banner()
        
        # Initialize configuration
        print("Initializing system configuration...")
        config = GuardConfig()
        print("✓ Configuration loaded")
        
        # Initialize Whisper ASR model
        print("\nLoading speech recognition model...")
        whisper_model = initialize_whisper(config)
        
        # ================================================================
        # MILESTONE 2: Face Recognition and Enrollment
        # ================================================================
        known_encodings, known_names = setup_face_enrollment(config)
        
        if not known_encodings:
            print("\n❌ Cannot proceed without enrolled faces. Exiting.")
            return
        
        # ================================================================
        # MILESTONE 1: Wait for Voice Activation
        # ================================================================
        print("\n" + "="*70)
        print("SYSTEM READY - WAITING FOR ACTIVATION")
        print("="*70)
        
        activated = wait_for_activation(whisper_model, config, activation_keyword="guard my room")
        
        if not activated:
            print("\n❌ System not activated. Exiting.")
            retry = input("Try activation again? (y/n): ").strip().lower()
            if retry == 'y':
                activated = wait_for_activation(whisper_model, config)
            
            if not activated:
                return
        
        # ================================================================
        # MILESTONE 3: Active Monitoring with Escalation
        # ================================================================
        print("\n" + "="*70)
        print("MILESTONE 3: GUARD MODE ACTIVATED")
        print("="*70)
        
        speak("Guard mode activated. I am now monitoring the room.")
        print("\nThe AI Guard is now active and will:")
        print("  1. Recognize trusted individuals")
        print("  2. Detect unknown persons")
        print("  3. Engage in escalating conversation with intruders")
        print("\nPress 'q' in the video window to deactivate.\n")
        
        input("Press Enter to start monitoring...")
        
        # Start main monitoring loop
        monitor_room(
            whisper_model=whisper_model,
            config=config,
            known_encodings=known_encodings,
            known_names=known_names,
            video_source=0
        )
        
        # Deactivation
        print("\n" + "="*70)
        print("GUARD MODE DEACTIVATED")
        print("="*70)
        speak("Guard mode deactivated. Have a good day.")
        
    except KeyboardInterrupt:
        print("\n\n⚠ System interrupted by user")
        speak("System shutting down.")
    
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        speak("System error occurred. Shutting down.")
    
    finally:
        print("\n" + "="*70)
        print("AI GUARD SYSTEM SHUTDOWN COMPLETE")
        print("="*70 + "\n")


def quick_test_mode():
    """Quick test mode - skips activation, goes straight to monitoring"""
    print("\n" + "="*70)
    print("QUICK TEST MODE - Skipping activation")
    print("="*70 + "\n")
    
    config = GuardConfig()
    whisper_model = initialize_whisper(config)
    known_encodings, known_names = load_face_encodings(config)
    
    if not known_encodings:
        print("❌ No enrolled faces. Please run normal mode first.")
        return
    
    print(f"✓ Loaded {len(known_names)} known face(s)")
    print("\nStarting monitoring...\n")
    
    monitor_room(
        whisper_model=whisper_model,
        config=config,
        known_encodings=known_encodings,
        known_names=known_names,
        video_source=0
    )


if __name__ == "__main__":
    # Check for test mode flag
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        quick_test_mode()
    else:
        main()
