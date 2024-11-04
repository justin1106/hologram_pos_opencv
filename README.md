# 홀로그램 POS기
## 1. 개발언어/라이브러리
* 개발 언어: Python
* 사용한 라이브러리: OpenCV, Mediapipe, Numpy, Math
## 2. 사용법
1. 테이블/버튼 위에 손가락을 가져간다.
2. 검지와 엄지를 붙인다.
3. 클릭이 된다.
## 3. 원리
* 손의 크기와 손가락 사이의 거리의 비율 구하기
    * Mediapipe와 Math 사용 
* 그 후 적절한 비율과 비교하기
    * 코드에서는 3
* 더 가깝다면 클릭으로 인식
```python
def calculating_hand_org():
    global click_state, real_length, hand_length, rate, results, frame_width
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            real_length = math.sqrt(pow(abs(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x*
                                            frame_width-hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x*frame_width), 2)+
                                            pow(abs(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y*frame_height
                                                         - hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y*frame_height), 2))
            hand_length = math.sqrt(pow(abs(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x*
                                            frame_width-hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x*frame_width), 2) + 
                                            pow(abs(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y*frame_height - 
                                                    hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y*frame_height), 2))
            rate = hand_length / real_length
            if rate>3:
                click_state = 1
                
            else:
                click_state = 0
```
