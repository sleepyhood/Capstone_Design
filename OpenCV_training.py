from PIL import Image
import numpy as np
import cv2
import os
import Varable as v

# ===============================수집


def dataCollect(data_path):
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video width
    cam.set(4, 480)  # set video height

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # For each person, enter one numeric face id
    face_id = int(input("\n enter user id end press <return> ==>  "))
    face_name = input("\n enter user name end press <return> ==>  ")
    print("\n [INFO] Initializing face capture. Look the camera and wait ...")

    # Initialize individual sampling face count
    count = 0
    # ===============================데이터 수집
    while True:
        ret, img = cam.read()
        # img = cv2.flip(img, -1) # flip video image vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            # Save the captured image into the datasets folder
            cv2.imwrite(
                str(data_path)
                + "/"
                + str(face_id)
                + "."
                + str(face_name)
                + "."
                + str(count)
                + ".jpg",
                img,
            )
            cv2.imshow("image", img)
        k = cv2.waitKey(100) & 0xFF  # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= v.collectConut:  # Take 30 face sample and stop video
            break
        print(count)
    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()
    return face_id, face_name


# ===============================학습


def training(data_path, training_path):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    # print(f"current face_id: {face_id}")

    def getImagesAndLabels(path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        # 모든 jpg들을 모아서 하나의 yml을 생성
        faceSamples = []
        ids = []
        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert("L")  # convert it to grayscale
            # print(f"PIL_img:{PIL_img}")
            img_numpy = np.array(PIL_img, "uint8")
            id = int(os.path.split(imagePath)[-1].split(".")[0])
            # print(id)
            faces = face_detector.detectMultiScale(img_numpy)
            for x, y, w, h in faces:
                faceSamples.append(img_numpy[y : y + h, x : x + w])
                ids.append(id)
        return faceSamples, ids

    print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces, ids = getImagesAndLabels(data_path)
    recognizer.train(faces, np.array(ids))

    # Save the model into trainer/trainer.yml
    # recognizer.save() worked on Mac, but not on Pi
    recognizer.write(str(training_path) + "/trainer" + ".yml")
    # Print the numer of faces trained and end program
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
