#홀로그램 키보드 (작동 오류 수정본)
import mediapipe as mp
import numpy as np
import cv2
import time
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

button_size = 160
gap = 30
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (0, 0, 0)
lineType = 2

PRESS_RATE = 3.0      # 이 값을 넘으면 클릭 시작
RELEASE_RATE = 2.5    # 이 값 아래로 내려가야 클릭 해제 (히스테리시스)
DONE_SECONDS = 3      # 결제 완료 화면 유지 시간(초)
DWELL_MODE = False    # True면 핀치 대신 머무르기(dwell) 방식으로 시작 (실행 중 d 키로 전환)
DWELL_SECONDS = 1.0   # 한 곳에 손가락을 이 시간만큼 두면 클릭

payment = ['10000', '20000', '30000', '40000', '50000', '60000']

menus = {
    '1': ['menu 1', 'menu 2'],
    '2': ['menu 1', 'menu 3'],
    '3': ['menu 2', 'menu 3'],
    '4': ['menu 4', 'menu 5'],
    '5': ['menu 3', 'menu 6'],
    '6': ['menu 3', 'menu 5']
}

selected = None        # 선택된 버튼 번호 (0~5), 없으면 None
state = False          # 결제 완료 화면 표시 중인지
paid_index = None      # 결제 중인 버튼 번호
paid_amount = ''       # 결제 중인 금액
done_start = 0.0       # 결제 완료 화면 시작 시각
click_state = 0        # 현재 핀치 상태
prev_click_state = 0   # 직전 프레임 핀치 상태
dwell_target = None    # 현재 손가락이 머무는 대상 (0~5, 'ok', None)
dwell_start = 0.0      # 그 대상에 머물기 시작한 시각
dwell_fired = False    # 이미 클릭이 발생했는지 (대상을 벗어나야 다시 가능)


def add_white_rectangle(frame, x, y, w, h):
    sub_img = frame[y:y+h, x:x+w]
    white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255

    res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)

    frame[y:y+h, x:x+w] = res
    return frame


def key_label(i):
    return str(i + 1) + ('V' if selected == i else '')


def pay(frame):
    if selected is None:
        return
    cv2.putText(frame, payment[selected], (55+button_size, button_size*2+gap*3+25), font, fontScale, fontColor, lineType)
    if payment[selected] != '0':
        cv2.putText(frame, 'OK', (2*(button_size+gap)+55, button_size*2+gap*3+25), font, fontScale, fontColor, lineType)


def draw_keyboard(frame):
    for x in range(3):
        frame = add_white_rectangle(frame, x*(button_size+gap)+50, gap, button_size, button_size)
        cv2.putText(frame, key_label(x), (x*(button_size+gap)+50, gap+25), font, fontScale, fontColor, lineType)
        for idx, m in enumerate(menus[str(x+1)]):
            cv2.putText(frame, m, (x*(button_size+gap)+60, gap+30+30*(idx+1)), font, fontScale, fontColor, lineType)
    for x in range(3):
        frame = add_white_rectangle(frame, x*(button_size+gap)+50, button_size+gap*2, button_size, button_size)
        cv2.putText(frame, key_label(x+3), (x*(button_size+gap)+50, 25+button_size+gap*2), font, fontScale, fontColor, lineType)
        for idx, m in enumerate(menus[str(x+4)]):
            cv2.putText(frame, m, (x*(button_size+gap)+50, 25+button_size+gap*2+30*(idx+1)), font, fontScale, fontColor, lineType)
    frame = add_white_rectangle(frame, 2*(button_size+gap)+50, button_size*2+gap*2+25, button_size//2, gap)
    cv2.putText(frame, 'total:', (100, gap*4-10 + button_size*2), font, fontScale, fontColor, lineType)
    add_white_rectangle(frame, 50 + button_size, gap*3+button_size*2, button_size+gap, gap)
    pay(frame)

    return frame


def calculating_hand_org():
    global click_state
    # 손이 없으면 클릭 상태 초기화
    if not results.multi_hand_landmarks:
        click_state = 0
        return
    lm = results.multi_hand_landmarks[0].landmark
    real_length = math.hypot(
        (lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].x - lm[mp_hands.HandLandmark.THUMB_TIP].x) * frame_width,
        (lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y - lm[mp_hands.HandLandmark.THUMB_TIP].y) * frame_height)
    hand_length = math.hypot(
        (lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x - lm[mp_hands.HandLandmark.WRIST].x) * frame_width,
        (lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y - lm[mp_hands.HandLandmark.WRIST].y) * frame_height)
    if real_length < 1:   # 0으로 나누기 방지
        real_length = 1
    rate = hand_length / real_length

    # 누를 때와 뗄 때 기준을 다르게 둬서 경계에서 떨리는 것 방지
    if click_state == 0 and rate > PRESS_RATE:
        click_state = 1
    elif click_state == 1 and rate < RELEASE_RATE:
        click_state = 0


def typing_calculate(hand_x, hand_y):
    global selected, state, paid_index, paid_amount, done_start
    if state:
        return

    row = None
    if hand_y >= gap and hand_y < button_size+gap:
        row = 0
    elif hand_y >= button_size+gap*2 and hand_y < button_size*2+gap*2:
        row = 1

    if row is not None:
        for x in range(3):
            if hand_x > x*(button_size+gap)+50 and hand_x < x*(button_size+gap)+50+button_size:
                selected = row*3 + x
    elif hand_y >= button_size*2+gap*2+25 and hand_y < button_size*2+gap*3+25:
        if hand_x > 2*(button_size+gap)+50 and hand_x < 2*(button_size+gap)+50+button_size//2:
            # 선택된 상품이 있고 아직 결제 안 된 경우에만 OK 동작
            if selected is not None and payment[selected] != '0':
                state = True
                paid_index = selected
                paid_amount = payment[selected]
                done_start = time.time()


def get_target(hand_x, hand_y):
    # 손가락 끝이 올라가 있는 대상 반환: 버튼 번호(0~5), 'ok', 또는 None
    row = None
    if hand_y >= gap and hand_y < button_size+gap:
        row = 0
    elif hand_y >= button_size+gap*2 and hand_y < button_size*2+gap*2:
        row = 1
    if row is not None:
        for x in range(3):
            if hand_x > x*(button_size+gap)+50 and hand_x < x*(button_size+gap)+50+button_size:
                return row*3 + x
        return None
    if hand_y >= button_size*2+gap*2+25 and hand_y < button_size*2+gap*3+25:
        if hand_x > 2*(button_size+gap)+50 and hand_x < 2*(button_size+gap)+50+button_size//2:
            if selected is not None and payment[selected] != '0':
                return 'ok'
    return None


def update_dwell(has_hand, hand_x, hand_y):
    # 진행도(0.0~1.0)를 반환하고, 가득 차면 클릭 처리
    global dwell_target, dwell_start, dwell_fired
    target = get_target(hand_x, hand_y) if (has_hand and not state) else None
    if target is None or target != dwell_target:
        dwell_target = target
        dwell_start = time.time()
        dwell_fired = False
        return 0.0
    if dwell_fired:
        return 0.0
    progress = (time.time() - dwell_start) / DWELL_SECONDS
    if progress >= 1.0:
        dwell_fired = True
        typing_calculate(hand_x, hand_y)
        return 0.0
    return progress


def done(c, idx):
    sub_img = frame[button_size+gap:button_size+gap+gap*3, button_size//2:button_size//2+frame_width-button_size]
    white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0

    res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)

    frame[button_size+gap:button_size+gap+gap*3, button_size//2:button_size//2+frame_width-button_size] = res

    cv2.putText(frame, f'{c}won payed ok!', (frame_width//2-200, frame_height//2), font, 1.3, (255, 255, 255), lineType)
    menus[str(idx+1)] = []


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    raise SystemExit

fail_count = 0
with mp_hands.Hands(model_complexity=0, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            fail_count += 1
            if fail_count > 30:
                print("카메라를 찾을 수 없습니다.")
                break
            continue
        fail_count = 0
        frame = cv2.flip(frame, 1)

        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)
        frame_height, frame_width, _ = frame.shape
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 키보드는 손 유무와 상관없이 항상 그림 (손이 위에 보이도록 먼저 그림)
        frame = draw_keyboard(frame)

        hand_x = 0
        hand_y = 0
        has_hand = bool(results.multi_hand_landmarks)
        if has_hand:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            hand_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x*frame_width
            hand_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y*frame_height

        if DWELL_MODE:
            # 한 곳에 일정 시간 머물면 클릭, 진행도를 손가락 끝에 원으로 표시
            progress = update_dwell(has_hand, hand_x, hand_y)
            if has_hand and progress > 0:
                cv2.ellipse(frame, (int(hand_x), int(hand_y)), (25, 25), -90, 0, 360*progress, (0, 255, 0), 4)
        else:
            calculating_hand_org()
            # 핀치를 시작한 순간(0 -> 1)에만 한 번 입력
            if has_hand and click_state == 1 and prev_click_state == 0:
                typing_calculate(hand_x, hand_y)
            prev_click_state = click_state

        cv2.putText(frame, 'mode: ' + ('dwell' if DWELL_MODE else 'pinch') + ' (d: switch)', (10, frame_height-10), font, 0.5, (255, 255, 255), 1)

        if state:
            if time.time() - done_start >= DONE_SECONDS:
                state = False
                payment[paid_index] = '0'
                selected = None
            else:
                done(paid_amount, paid_index)

        cv2.imshow('frame', frame)
        key = cv2.waitKey(5) & 0xFF
        if key == 27:
            break
        elif key == ord('d'):
            # 클릭 방식 전환, 상태 초기화해서 전환 직후 오입력 방지
            DWELL_MODE = not DWELL_MODE
            click_state = 0
            prev_click_state = 0
            dwell_target = None
            dwell_fired = False
cap.release()
cv2.destroyAllWindows()
