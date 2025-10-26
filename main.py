import logging
import os
import sys
import time
import json
import whisper
import torch
from pathlib import Path
import csv

import yt_dlp


def create_transcription_file(transcribed_text, filename, transcription_output_folder):

    name_without_extension = Path(filename).stem        # Remove the extension (.m4a)

    with open(f'{transcription_output_folder}\\{name_without_extension}.txt', 'w', encoding="utf-8") as file:
        file.write(transcribed_text)

def transcribe_audio(input_audio_file, cuda_device):

    print(f'Running audio conversion for: {os.path.basename(input_audio_file)}')

    model = whisper.load_model("base", device=cuda_device)
    result = model.transcribe(input_audio_file)
    transcribed_text = result['text']

    return transcribed_text

def transcribe_and_output_text(audio_file_folder, audio_file, cuda_device, transcription_output_folder):

    transcribed_text = transcribe_audio(str(os.path.join(audio_file_folder, audio_file)), cuda_device)
    # After doing the transcription, make the .txt file with the text transcription.
    create_transcription_file(transcribed_text, audio_file, transcription_output_folder)

def read_settings_file():

    with open("settings.json", "r") as f:
        settings_json = json.load(f)

    audio_file_folder = settings_json['input_audio']
    transcription_output_folder = settings_json['output_transcriptions']
    log_file = settings_json['log_file']

    return audio_file_folder, transcription_output_folder, log_file

def generate_time(start_time):

    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time  # Calculate elapsed time
    # Convert seconds to mm:ss format
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    return minutes, seconds

def main():

    # ----- Check if CUDA is available
    cuda_device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----- Extract the JSON settings -----

    audio_file_folder, transcription_output_folder, log_file = read_settings_file()

    # ----- Check if the log file exists and is not empty so that the headers aren't written again -----
    file_exists = os.path.isfile(log_file) and os.path.getsize(log_file) > 0

    application_running = True

    while application_running:

        print("Level 1:")
        print("1: List audio files.")
        print("2: Transcribe audio files.")
        print("3: Exit")
        print("4: Test yt-dlp")
        user_input = input("Enter an option: ")
        print()

        if user_input == "1":

            items = os.listdir(audio_file_folder)
            for item in items:
                print(item)

            print()

        elif user_input == "2":   # Transcribe audio files

            with open(log_file, "a", newline="") as file:

                writer = csv.writer(file)       # Load the .csv file into memory.

                # Write headers only if the file does not exist or is empty
                if not file_exists:
                    writer.writerow(["elapsed_time", "audio_file"])

                # Cycle through the folder of audio files
                for audio_file in os.listdir(audio_file_folder):

                    start_time = time.time()  # Start timing

                    transcribe_and_output_text(audio_file_folder, audio_file, cuda_device, transcription_output_folder)

                    minutes, seconds = generate_time(start_time)

                    processing_message = f"Time for processing '{audio_file}': {minutes:02}:{seconds:02}"
                    processing_message_list = [f"{minutes:02}:{seconds:02}", f"{audio_file}"]

                    print(processing_message + "\n")

                    writer.writerow(processing_message_list)

            print()

        elif user_input == "3":

            sys.exit()

        elif user_input == "4":

            URLS = ["https://www.youtube.com/watch?v=UQ2XkN2EG1M"]

            ydl_opts = {
                'format': 'm4a/bestaudio/best',
                # See help(yt_dlp.postprocessor) for a list of available Postprocessors and their arguments
                'postprocessors': [{  # Extract audio using ffmpeg
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                }]
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                error_code = ydl.download(URLS)
                logging.log(0, "Error code:", error_code)

if __name__ == '__main__':

    main()
