from flask import Flask, render_template, Response
from threading import Thread
import speech_recognition as sr
from google.cloud import speech
import google_stream_stt as gs
import pyaudio
import sys

app = Flask(__name__)
transcript_result = ""


# 이는 데모 버전
# googlestt()가 무한반복됨
def googleSTT():
    global transcript_result

    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="ko-KR",
        max_alternatives=1,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    mic_manager = gs.ResumableMicrophoneStream(16000, 1024)
    sys.stdout.write("\nListening, say 'Quit' or 'Exit' to stop.\n\n")
    sys.stdout.write("End (ms)       Transcript Results/Status\n")
    sys.stdout.write("=====================================================\n")

    with mic_manager as stream:
        while not stream.closed:
            stream.audio_input = []
            audio_generator = stream.generator()

            requests = (
                speech.StreamingRecognizeRequest(audio_content=content)
                for content in audio_generator
            )

            responses = client.streaming_recognize(streaming_config, requests)
            transcript_result = gs.listen_print_loop(responses, stream)

            print(transcript_result)

            if stream.result_end_time > 0:
                stream.final_request_end_time = stream.is_final_end_time
            stream.result_end_time = 0
            stream.last_audio_input = []
            stream.last_audio_input = stream.audio_input
            stream.audio_input = []
            stream.restart_counter = stream.restart_counter + 1

            if not stream.last_transcript_was_final:
                sys.stdout.write("\n")
            stream.new_stream = True


def start_google_stt():
    global transcript_result
    while True:
        googleSTT()
        # Response(transcript_result, content_type="text/event-stream")


@app.route("/")
def index():
    return render_template("index2.html")


@app.route("/transcript")
def stream():
    return Response(transcript_result, content_type="text/event-stream")


if __name__ == "__main__":
    # Start a separate thread for Google STT
    stt_thread = Thread(target=start_google_stt)
    stt_thread.start()

    # Run the Flask app
    app.run(debug=True, threaded=True)
