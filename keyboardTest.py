import keyboard


def main():
    while True:
        # q 또는 esc 키를 누르면 반복문 종료
        print("q 또는 esc 키를 누르면 반복문 종료")

        if keyboard.is_pressed("q") or keyboard.is_pressed("esc"):
            print("프로그램을 종료합니다.")
            break

        # 여기에서 수행할 작업을 추가


if __name__ == "__main__":
    main()
