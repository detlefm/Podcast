import asyncio
import os
from datetime import datetime
try:
    from src.paudio import create_podcast_audio
    from src.utils.utils import get_last_timestamp, add_feedback_to_state, OUT_FOLDER
    from src.utils.textGDwithWeightClipping import optimize_prompt
except ImportError:
    from paudio import create_podcast_audio
    from utils.utils import get_last_timestamp, add_feedback_to_state, OUT_FOLDER
    from utils.textGDwithWeightClipping import optimize_prompt

async def create_podcast_with_feedback(pdf_path, timestamp=None):
    # Get the last timestamp if not provided or if 'last' is specified
    if timestamp is None or timestamp.lower() == "last":
        timestamp = get_last_timestamp()
    
    print(f"Creating podcast using timestamp: {timestamp}")
    
    # Read the PDF file as bytes
    with open(pdf_path, 'rb') as pdf_file:
        pdf_content = pdf_file.read()
    
    # Create the podcast audio
    audio_bytes, dialogue_text, new_timestamp = await create_podcast_audio(
        pdf_content,
        timestamp=timestamp,
        summarizer_model="gpt-4o",
        scriptwriter_model="gpt-4o",
        enhancer_model="gpt-4o",
        provider="OpenAI"
    )
    
    print("\nPodcast created successfully!")
    print(f"New timestamp: {new_timestamp}")
    
    # Save the audio file
    os.makedirs(os.path.join(OUT_FOLDER, "audios"), exist_ok=True)
    audio_filename = os.path.join(OUT_FOLDER, "audios", f"podcast_{new_timestamp}.mp3")
    with open(audio_filename, "wb") as audio_file:
        audio_file.write(audio_bytes)
    print(f"Audio saved as: {audio_filename}")
    
    # Save the dialogue text
    dialogue_filename = os.path.join(OUT_FOLDER, "audios", f"dialogue_{new_timestamp}.txt")
    with open(dialogue_filename, "w", encoding="utf-8") as dialogue_file:
        dialogue_file.write(dialogue_text)
    print(f"Dialogue saved as: {dialogue_filename}")
    
    # Print the dialogue
    print("\nGenerated Dialogue:")
    print(dialogue_text)
    
    # Ask for feedback
    want_feedback = input("\nDo you want to provide feedback? (yes/no): ").lower()
    
    if want_feedback.startswith("y"):
        feedback = input("Please provide your feedback: ")
        add_feedback_to_state(new_timestamp, feedback)
        print("Feedback added to the podcast state.")
        
        # Optimize prompts
        print("\nOptimizing prompts based on feedback...")
        optimize_prompt("summarizer", timestamp, new_timestamp, "gpt-4o", "gpt-4o")
        optimize_prompt("scriptwriter", timestamp, new_timestamp, "gpt-4o", "gpt-4o")
        optimize_prompt("enhancer", timestamp, new_timestamp, "gpt-4o", "gpt-4o")
        print("Prompts optimized successfully.")
    else:
        print("No feedback provided. Prompts will not be optimized.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create a podcast with feedback from a PDF file.")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--timestamp", help="Timestamp to use for prompts (format: YYYYMMDD_HHMMSS, or 'last' for the most recent). If not provided, uses the most recent.")
    args = parser.parse_args()
    
    asyncio.run(create_podcast_with_feedback(args.pdf_path, args.timestamp))
