from flask import Flask, render_template, request, redirect
import speech_recognition as sr
import Varable as v

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index2.html", transcript=transcript)


r = sr.Recognizer()

with sr.Microphone() as source:
    print("Say something!")
    audio = r.listen(source)
    transcript = 0
    index()
    # while True:
    try:
        transcript = r.recognize_google(audio, language="ko-KR")
        print("Google Speech Recognition thinks you said " + transcript)

    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(
            "Could not request results from Google Speech Recognition service; {0}".format(
                e
            )
        )

"""
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if __name__ == "__main__":
        app.run(debug=True, threaded=True)

"""
if __name__ == "__main__":
    app.run(debug=True, threaded=True)
