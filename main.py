"""
AI Guard Agent - Main Entry Point
==================================
EE782 Programming Assignment 2

This is the main orchestrator file for the AI Guard system.
It coordinates all components and manages the program flow.

The system follows this milestone structure:
- Milestone 1: Voice activation via speech command
- Milestone 2: Face recognition and enrollment of trusted persons
- Milestone 3: Intelligent escalation dialogue with unknown persons

Project Components:
- ASR (Automatic Speech Recognition): Whisper by OpenAI
- TTS (Text-to-Speech): Google TTS
- Face Recognition: face_recognition library (based on dlib)
- LLM (Large Language Model): Google Gemini for intelligent conversation

Usage:
    python main.py              # Normal mode (full workflow)
    python main.py --test       # Quick test mode (skip activation)
"""

# ============================================================================
# IMPORTS
# ============================================================================

import sys  # For command-line arguments and system functions
import os   # For operating system operations

# Import all necessary functions from guard_functions module
from guard_functions import (
    GuardConfig,                 # Configuration class
    initialize_whisper,          # Initialize ASR model
    listen_for_command,          # Listen for voice activation
    load_face_encodings,         # Load saved face data
    save_face_encodings,         # Save face data to disk
    enroll_face_from_webcam,     # Enroll new trusted person
    monitor_room,                # Main monitoring loop
    speak,                       # Text-to-speech function
    initialize_llm               # Initialize conversational AI
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_banner():
    """
    Print welcome banner to console
    
    This function displays an attractive ASCII banner when the program starts,
    providing visual clarity and branding for the application.
    """
    print("\n" + "="*70)
    print("    AI GUARD AGENT - INTELLIGENT ROOM MONITORING SYSTEM")
    print("    EE782: Advanced Topics in Machine Learning")
    print("="*70 + "\n")


def system_setup(config):
    """
    Setup phase: Initialize all AI components
    
    This function handles the initialization of all major system components:
    1. Whisper ASR model for speech recognition
    2. Gemini LLM for intelligent conversation
    
    This is done once at startup to avoid repeated loading delays.
    
    Args:
        config: GuardConfig object containing all settings
    
    Returns:
        tuple: (whisper_model, llm_client)
            - whisper_model: Loaded WhisperModel instance for ASR
            - llm_client: Gemini model instance for conversation
    """
    print("-"*70)
    print("SYSTEM SETUP")
    print("-"*70)
    
    # Display configuration status
    print("Initializing system configuration...")
    print("✓ Configuration loaded")
    
    # Initialize Whisper ASR model
    # This loads the pre-trained model into memory
    print("\nLoading speech recognition model...")
    whisper_model = initialize_whisper(config)
    
    # Initialize Large Language Model (Gemini)
    # This configures the API connection for intelligent conversation
    print("\nInitializing conversational AI...")
    llm_client = initialize_llm(config)
    
    return whisper_model, llm_client


def setup_face_enrollment(config):
    """
    Face Recognition Setup and Enrollment
    
    This function manages the face enrollment process, which is crucial
    for the system to distinguish between trusted and unknown persons.
    
    The function:
    1. Attempts to load existing face encodings from disk
    2. Displays list of enrolled persons if any exist
    3. Allows user to add additional trusted persons
    4. Ensures at least one person is enrolled before proceeding
    
    Face Encoding Process:
    - Captures multiple images of a person's face
    - Extracts 128-dimensional feature vectors (encodings)
    - Averages multiple encodings for robustness
    - Saves to disk for persistence
    
    Args:
        config: GuardConfig object
    
    Returns:
        tuple: (known_encodings, known_names)
            - known_encodings: List of numpy arrays (face encodings)
            - known_names: List of strings (corresponding names)
    """
    # Try to load existing encodings from disk
    known_encodings, known_names = load_face_encodings(config)
    
    # If we found existing enrolled faces
    if known_encodings:
        print(f"\nFound {len(known_names)} enrolled face(s):")
        
        # Display list of enrolled persons
        for i, name in enumerate(known_names, 1):
            print(f"  {i}. {name}")
        
        # Ask if user wants to add more trusted persons
        while True:
            choice = input("\nAdd another trusted person? (y/n): ").strip().lower()
            
            if choice == 'y':
                # User wants to add another person
                name = input("Enter name: ").strip()
                
                if name:
                    # Enroll the new person via webcam
                    # This captures 5 samples by default
                    encoding = enroll_face_from_webcam(config, name, num_samples=5)
                    
                    if encoding is not None:
                        # Successfully enrolled, add to lists
                        known_encodings.append(encoding)
                        known_names.append(name)
                        
                        # Save updated encodings to disk
                        save_face_encodings(config, known_encodings, known_names)
            
            elif choice == 'n':
                # User is done adding persons
                break
            
            else:
                print("Please enter 'y' or 'n'")
    
    else:
        # No existing faces found - must enroll at least one
        print("\nNo trusted faces found. You must enroll at least one person.")
        
        # Continue asking until at least one person is enrolled
        while not known_encodings:
            name = input("\nEnter name of trusted person: ").strip()
            
            if name:
                # Attempt enrollment
                encoding = enroll_face_from_webcam(config, name, num_samples=5)
                
                if encoding is not None:
                    # Successfully enrolled
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
    Wait for voice activation command (Milestone 1)
    
    This function implements the voice activation feature where the system
    remains idle until the user speaks a specific command phrase.
    
    How it works:
    1. System announces it's ready and waiting
    2. Records audio when user is ready
    3. Transcribes audio using Whisper
    4. Checks if activation keyword is present
    5. Repeats up to 3 times if keyword not detected
    
    This provides a natural, hands-free way to activate the guard system.
    
    Args:
        whisper_model: Loaded WhisperModel instance for ASR
        config: GuardConfig object
        activation_keyword: Phrase that activates guard mode (default: "guard my room")
    
    Returns:
        bool: True if activation successful, False if failed after max attempts
    """
    print("\n" + "="*70)
    print("SYSTEM READY - WAITING FOR ACTIVATION")
    print("="*70)
    
    # Announce readiness via TTS
    speak("Do you want me to start guarding the room? Please say the command if you do.")
    print("\nThe system will listen for this command to enter guard mode.\n")
    
    # Allow multiple attempts to activate
    max_attempts = 3
    attempt = 0
    
    while attempt < max_attempts:
        attempt += 1
        print(f"Attempt {attempt}/{max_attempts}")
        
        # Listen for the activation command
        if listen_for_command(whisper_model, config, activation_keyword):
            # Activation successful!
            return True
        else:
            # Keyword not detected
            print("Activation keyword not detected.")
            if attempt < max_attempts:
                print("Please try again...\n")
    
    # Failed to activate after all attempts
    print(f"\nFailed to activate after {max_attempts} attempts.")
    return False


# ============================================================================
# MAIN APPLICATION FUNCTION
# ============================================================================

def main():
    """
    Main function - orchestrates the entire AI Guard system
    
    This is the primary entry point that coordinates all system components
    and manages the complete workflow from startup to shutdown.
    
    Program Flow:
    1. Display welcome banner
    2. Initialize configuration
    3. Setup AI components (Whisper, Gemini)
    4. Enroll/load trusted faces
    5. Wait for voice activation
    6. Start room monitoring
    7. Handle known/unknown persons
    8. Clean shutdown
    
    The main loop allows returning to enrollment after monitoring,
    enabling dynamic addition of trusted persons without restarting.
    """
    
    try:
        # ====================================================================
        # INITIALIZATION PHASE
        # ====================================================================
        
        # Print welcome banner
        print_banner()
        
        # Initialize configuration object
        # This creates directories and sets all parameters
        config = GuardConfig()
        
        # ====================================================================
        # MAIN LOOP
        # ====================================================================
        # This loop allows returning to enrollment after monitoring
        
        while True:
            # ================================================================
            # PHASE 1: System Setup
            # ================================================================
            # Initialize all AI models (Whisper ASR, Gemini LLM)
            whisper_model, llm_client = system_setup(config)
            
            # ================================================================
            # PHASE 2: Face Recognition and Enrollment
            # ================================================================
            # Load existing or enroll new trusted persons
            known_encodings, known_names = setup_face_enrollment(config)
            
            # Verify we have at least one enrolled face
            if not known_encodings:
                print("\nCannot proceed without enrolled faces. Exiting.")
                return
            
            # ================================================================
            # PHASE 3: Voice Activation (Milestone 1)
            # ================================================================
            # Wait for user to speak activation command
            activated = wait_for_activation(whisper_model, config, activation_keyword="guard my room")
            
            # Check if activation was successful
            if not activated:
                print("\nSystem not activated. Exiting.")
                
                # Give user option to retry
                retry = input("Try activation again? (y/n): ").strip().lower()
                if retry == 'y':
                    continue  # Go back to beginning of loop (enrollment phase)
                else:
                    return  # Exit program
            
            # ================================================================
            # PHASE 4: Guard Mode Active
            # ================================================================
            print("\n" + "="*70)
            print("GUARD MODE ACTIVATED")
            print("="*70 + "\n")
            
            # Announce activation via TTS
            speak("Guard mode activated. I am now monitoring the room.")
            
            # Display capabilities
            print("The AI Guard is now active and will:")
            print("  1. Recognize trusted individuals")
            print("  2. Detect unknown persons")
            print("  3. Engage in escalating conversation with intruders")
            
            # Wait for user confirmation before starting monitoring
            input("Press Enter to start monitoring...")
            
            # ================================================================
            # PHASE 5: Main Monitoring Loop
            # ================================================================
            # This is the core function that:
            # - Captures video from webcam
            # - Recognizes faces in real-time
            # - Handles unknown persons with conversation
            # - Returns when known person detected or user quits
            
            should_continue = monitor_room(
                whisper_model=whisper_model,
                llm_client=llm_client,
                config=config,
                known_encodings=known_encodings,
                known_names=known_names,
                video_source=0  # Use default webcam
            )
            
            # Check if we should continue or exit
            if not should_continue:
                # User pressed 'q' to quit completely
                break
            
            # If should_continue is True, loop continues and returns to enrollment
            # This happens when a known face is detected
        
        # ====================================================================
        # SHUTDOWN PHASE
        # ====================================================================
        print("\n" + "="*70)
        print("GUARD MODE DEACTIVATED")
        print("="*70)
        
        # Announce shutdown via TTS
        speak("Guard mode deactivated. Have a good day.")
        
    # ========================================================================
    # ERROR HANDLING
    # ========================================================================
    
    except KeyboardInterrupt:
        # User pressed Ctrl+C
        print("\n\nSystem interrupted by user")
        speak("System shutting down.")
    
    except Exception as e:
        # Unexpected error occurred
        print(f"\nFatal error: {e}")
        
        # Print detailed traceback for debugging
        import traceback
        traceback.print_exc()
        
        # Announce error via TTS
        speak("System error occurred. Shutting down.")
    
    finally:
        # This always executes, even if there's an error
        # Ensures clean shutdown message
        print("\n" + "="*70)
        print("AI GUARD SYSTEM SHUTDOWN COMPLETE")
        print("="*70 + "\n")


# ============================================================================
# QUICK TEST MODE
# ============================================================================

def quick_test_mode():
    """
    Quick test mode - skips activation, goes straight to monitoring
    
    This is a convenience function for testing the monitoring system
    without going through the full activation process each time.
    
    Useful during development and debugging.
    
    Usage:
        python main.py --test
    """
    print("\n" + "="*70)
    print("QUICK TEST MODE - Skipping activation")
    print("="*70 + "\n")
    
    # Initialize configuration
    config = GuardConfig()
    
    # Setup AI components
    whisper_model, llm_client = system_setup(config)
    
    # Load existing face encodings (don't allow enrollment in test mode)
    known_encodings, known_names = load_face_encodings(config)
    
    # Verify we have enrolled faces
    if not known_encodings:
        print("No enrolled faces. Please run normal mode first.")
        return
    
    print(f"✓ Loaded {len(known_names)} known face(s)")
    print("\nStarting monitoring...\n")
    
    # Start monitoring immediately
    monitor_room(
        whisper_model=whisper_model,
        llm_client=llm_client,
        config=config,
        known_encodings=known_encodings,
        known_names=known_names,
        video_source=0
    )


# ============================================================================
# PROGRAM ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Entry point when script is run directly
    
    Checks for command-line arguments:
    - No arguments: Run normal mode (full workflow)
    - --test: Run quick test mode (skip activation)
    """
    
    # Check for test mode flag
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Quick test mode requested
        quick_test_mode()
    else:
        # Normal mode
        main()
