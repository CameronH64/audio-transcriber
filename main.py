import os
import threading
import time
import json
import whisper
import torch
from pathlib import Path
import csv


def create_transcription_file(transcribed_text, filename, transcription_output_folder):

    name_without_extension = Path(filename).stem        # Remove the extension (.m4a)

    with open(f'{transcription_output_folder}\\{name_without_extension}.txt', 'w', encoding="utf-8") as file:
        file.write(transcribed_text)


def paragraphize_text(input_string):

    # Convert the input string to a list of characters for easier manipulation
    chars = list(input_string)
    space_count = 0

    # Iterate over the characters and replace every 100th space with a newline
    for i in range(len(chars)):
        if chars[i] == ' ':
            space_count += 1
            if space_count % 100 == 0:
                chars[i] = '\n\n'

    # Join the list back into a string
    result_string = ''.join(chars)
    return result_string


def transcribe_audio(input_audio_file: str, cuda_device):

    print(f'Running audio conversion for: {os.path.basename(input_audio_file)}')

    model = whisper.load_model('base', device=cuda_device)
    result = model.transcribe(input_audio_file, fp16=False)
    transcribed_text = result['text']

    final_text = paragraphize_text(transcribed_text)

    return final_text


def transcribe_and_output_text(audio_file_folder, audio_file, cuda_device, transcription_output_folder):

    transcribed_text = transcribe_audio(os.path.join(audio_file_folder, audio_file), cuda_device)
    # After doing the transcription, make the .txt file with the text transcription.
    create_transcription_file(transcribed_text, audio_file, transcription_output_folder)


def read_settings_file():

    with open("settings.json", "r") as f:
        settings_json = json.load(f)

    return settings_json


def main():

    # Check if CUDA is available
    cuda_device = "cuda" if torch.cuda.is_available() else "cpu"

    settings_json = read_settings_file()

    audio_file_folder = settings_json['input_audio']
    transcription_output_folder = settings_json['output_transcriptions']
    log_file = settings_json['log_file']

    # Check if the log file exists and is not empty so that the headers aren't written again.
    file_exists = os.path.isfile(log_file) and os.path.getsize(log_file) > 0

    with open(log_file, "a", newline="") as file:

        writer = csv.writer(file)

        # Write headers only if the file does not exist or is empty
        if not file_exists:
            writer.writerow(["elapsed_time", "audio_file"])

        # Cycle through the folder of audio files
        for audio_file in os.listdir(audio_file_folder):

            start_time = time.time()  # Start timing

            transcribe_and_output_text(audio_file_folder, audio_file, cuda_device, transcription_output_folder)

            end_time = time.time()  # End timing
            elapsed_time = end_time - start_time  # Calculate elapsed time

            # Convert seconds to mm:ss format
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)

            processing_message = f"Time for processing '{audio_file}': {minutes:02}:{seconds:02}"
            processing_message_list = [f"{minutes:02}:{seconds:02}", f"{audio_file}"]

            print(processing_message + "\n")

            writer.writerow(processing_message_list)


if __name__ == '__main__':

    main()
