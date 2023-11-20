from PIL import Image
import numpy as np
import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height

# Path for face image database
data_path = "[OpenCV]dataset"
training_path = "[OpenCV]trainer"

# 딕셔너리 선언
face_dict = {
    -1: "None",
}


# 현재 저장된 yml 파일 불러오기
def list_files(folder_path):
    files = os.listdir(folder_path)
    print(f"\n{len(files)} Files in the folder:")
    # print("Files in the folder:")
    for file in files:
        print(file)
        key = int(file.split(".")[1])  # 파일명에서 키 추출
        value = file.split(".")[2]  # 파일명에서 값 추출
        face_dict[key] = value


print(list_files(training_path))
print(face_dict)
# before
# face_detector = cv2.CascadeClassifier(
#     'haarcascades/haarcascade_frontalface_default.xml')

# After
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# For each person, enter one numeric face id
face_id = int(input("\n enter user id end press <return> ==>  "))
face_name = input("\n enter user name end press <return> ==>  ")
print("\n [INFO] Initializing face capture. Look the camera and wait ...")


face_dict[face_id] = face_name
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
            str(data_path) + "/User." + str(face_id) + "." + str(count) + ".jpg", img
        )
        cv2.imshow("image", img)
    k = cv2.waitKey(100) & 0xFF  # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 30:  # Take 30 face sample and stop video
        break
    print(count)
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()


# ===============================학습


recognizer = cv2.face.LBPHFaceRecognizer_create()
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert("L")  # convert it to grayscale
        img_numpy = np.array(PIL_img, "uint8")
        id = int(os.path.split(imagePath)[-1].split(".")[1])
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
recognizer.write(
    str(training_path) + "/trainer." + str(face_id) + "." + str(face_name) + ".yml"
)
# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))


# ===============================예측
print(face_dict)

# recognizer = cv2.face.LBPHFaceRecognizer_create()
# before
# recognizer.read("#trainer/trainer.yml")
# 디렉토리 경로
# directory = "#trainer/"  # 실제 디렉토리 경로로 대체해야 합니다

# 모든 yml 파일 불러오기
for filename in os.listdir(training_path):
    if filename.endswith(".yml"):
        file_path = os.path.join(training_path, filename)
        recognizer.read(file_path)


face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
font = cv2.FONT_HERSHEY_SIMPLEX

# iniciate id counter
id = 0

# names related to ids: example ==> loze: id=1,  etc
# 이런식으로 사용자의 이름을 사용자 수만큼 추가해준다.
# names = ["None", "osw", "ljy", "chs", "ksw"]

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video widht
cam.set(4, 480)  # set video height

# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:
    ret, img = cam.read()
    # img = cv2.flip(img, -1)  # Flip vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y : y + h, x : x + w])
        # Check if confidence is less them 100 ==> "0" is perfect match

        if confidence < 100 and id in face_dict:
            id = face_dict[id]
            confidence = "  {0}%".format(round(100 - confidence))

        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
            roi = img[y : y + h, x : x + w]
            # cv2.GaussianBlur(roi, (가로 불투명도, 세로 불투명도), 0)
            blurred_roi = cv2.GaussianBlur(roi, (75, 75), 0)
            img[y : y + h, x : x + w] = blurred_roi

        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (0, 0, 0), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow("camera", img)
    k = cv2.waitKey(10) & 0xFF  # Press 'ESC' for exiting video
    if k == 27:
        break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
