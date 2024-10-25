import os
import threading
from concurrent.futures.thread import ThreadPoolExecutor

import whisper


def create_file(transcribed_text, filename, transcription_output_folder):

    with open(f'{transcription_output_folder}\\{filename}_transcribed.txt', 'w') as file:
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


def transcribe_audio(input_audio_file: str):

    print(f'Running audio conversion for {input_audio_file}')

    model = whisper.load_model('base')

    result = model.transcribe(input_audio_file, fp16=False)
    transcribed_text = result['text']

    print(transcribed_text)

    final_text = paragraphize_text(transcribed_text)

    return final_text


def transcribe_and_output_text(audio_file_folder, audio_file, transcription_output_folder):
    transcribed_text = transcribe_audio(os.path.join(audio_file_folder, audio_file))
    # After doing the transcription, make the .txt file with the text transcription.
    create_file(transcribed_text, audio_file, transcription_output_folder)
    print()


def transcribe_and_output_text_thread(audio_file_folder, audio_file, transcription_output_folder):

    my_thread = threading.Thread(target=transcribe_and_output_text, args=(audio_file_folder, audio_file, transcription_output_folder))
    my_thread.start()


def main():
    """Docstring here"""

    audio_file_folder = 'C:\\Users\\gold1\\pycharm-projects\\youtube-audio-transcriber-working-directory\\input_audio'
    transcription_output_folder = 'C:\\Users\\gold1\\pycharm-projects\\youtube-audio-transcriber-working-directory\\output_transcriptions'

    # Set a limit on the number of threads you want to run concurrently
    max_threads = 3  # You can adjust this number based on your RAM constraints

    # Use ThreadPoolExecutor to manage threads
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = []

        # Cycle through the folder of audio files
        for audio_file in os.listdir(audio_file_folder):
            # Submit each transcription task to the thread pool
            future = executor.submit(transcribe_and_output_text_thread, audio_file_folder, audio_file,
                                     transcription_output_folder)
            futures.append(future)
            # Optionally, wait for all threads to complete
            # for future in as_completed(futures):
            #     result = future.result()  # This will block until the individual task is done
            #     print(result)



if __name__ == '__main__':

    main()
