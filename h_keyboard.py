#홀로그램 키보드
import mediapipe as mp
import numpy as np
import cv2
import time
import math
import pyautogui
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

thickness = 1
click_state = 0
click_location = 0
on_focus = False
real_length = None
hand_length = None
rate = None
click = [False,False,False,False,False,False]
state = False
n = 0
button_size = 160
gap = 30
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (0,0,0)
lineType = 2
key_value_1 = ['1','2','3']
key_value_2 = ['4','5','6']
payment = ['10000','20000','30000','40000','50000','60000']

menus = {
    '1':['menu 1','menu 2'],
    '2':['menu 1','menu 3'],
    '3':['menu 2','menu 3'],
    '4':['menu 4','menu 5'],
    '5':['menu 3','menu 6'],
    '6':['menu 3','menu 5']
}
c_pay = ''

def add_white_rectangle(frame,x,y,w,h):
    sub_img = frame[y:y+h,x:x+w]
    white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255

    res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5,1.0)

    frame[y:y+h, x:x+w] = res
    return frame 
def pay():
    global c_pay,n
    for idx,k in enumerate(key_value_1):
        if len(k) == 2:
            cv2.putText(frame, payment[idx], (55+button_size,button_size*2+gap*3+25),font,fontScale,fontColor, lineType)
            cv2.putText(frame, 'OK', (2*(button_size+gap)+55,button_size*2+gap*3+25),font,fontScale,fontColor, lineType)
            c_pay = payment[idx]
            n = str(idx+1)
    for idx,k in enumerate(key_value_2):
        if len(k) == 2:
            cv2.putText(frame, payment[idx+3], (55+button_size,button_size*2+gap*3+25),font,fontScale,fontColor, lineType)
            cv2.putText(frame, 'OK', (2*(button_size+gap)+55,button_size*2+gap*3+25),font,fontScale,fontColor, lineType)
            c_pay = payment[idx+3]
            n = str(idx+4)
def draw_keyboard(frame):
    for x in range(3):
        frame = add_white_rectangle(frame,x*(button_size+gap)+50, gap, button_size, button_size)
        cv2.putText(frame, f'{key_value_1[x]}', (x*(button_size+gap)+50,gap+25),font,fontScale,fontColor, lineType)
        for idx,m in enumerate(menus[str(x+1)]):
            cv2.putText(frame, m, (x*(button_size+gap)+60,gap+30+30*(idx+1)),font,fontScale,fontColor,lineType)  
    for x in range(3):
        frame = add_white_rectangle(frame, x*(button_size+gap)+50,button_size+gap*2, button_size, button_size)
        cv2.putText(frame, f'{key_value_2[x]}', (x*(button_size+gap)+50,25+button_size+gap*2),font,fontScale,fontColor, lineType)
        for idx,m in enumerate(menus[str(x+4)]):
            cv2.putText(frame, m, (x*(button_size+gap)+50,25+button_size+gap*2+30*(idx+1)),font,fontScale,fontColor,lineType)  
    frame = add_white_rectangle(frame, 2*(button_size+gap)+50, button_size*2+gap*2+25, button_size//2, gap)
    cv2.putText(frame, 'total:', (100, gap*4-10 + button_size*2), font,fontScale,fontColor, lineType)
    add_white_rectangle(frame, 50 +button_size, gap*3+button_size*2,button_size+gap,gap)
    pay()
    
    return frame
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
def typing_calculate(hand_x,hand_y):
    global click_location, typing_text, back_space_state, enter_state, state
    if not state:
        if hand_y >= gap and hand_y < button_size+gap:
            for x in range(3):
                if hand_x > x*(button_size+gap)+50 and hand_x < x*(button_size+gap)+50+button_size:
                    for idx,k in enumerate(key_value_1):
                        if len(k) !=1:
                            key_value_1[idx] = str(idx+1)
                    for idx,k in enumerate(key_value_2):
                        if len(k) !=1:
                            key_value_2[idx] = str(idx+4)
                    key_value_1[x] = key_value_1[x]+'V'
        
        elif hand_y >= button_size+gap*2 and hand_y < button_size*2+gap*2:
            for x in range(3):
                if hand_x > x*(button_size+gap)+50 and hand_x < x*(button_size+gap)+50+button_size:
                    for idx,k in enumerate(key_value_2):
                        if len(k) !=1:
                            key_value_2[idx] = str(idx+4)
                    for idx,k in enumerate(key_value_1):
                        if len(k) !=1:
                            key_value_1[idx] = str(idx+1)
                    key_value_2[x] = key_value_2[x]+'V'
        elif hand_y >= button_size*2+gap*2+25 and hand_y<button_size*2+gap*3+25:
            if hand_x>2*(button_size+gap)+50 and hand_x < 2*(button_size+gap)+50+button_size//2:
                if c_pay !='0':
                    state = True
                        
def done(c,n):
    global key_value_1, key_value_2,state
    sub_img = frame[button_size+gap:button_size+gap+gap*3,button_size//2:button_size//2+frame_width-button_size]
    white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0

    res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5,1.0)

    frame[button_size+gap:button_size+gap+gap*3, button_size//2:button_size//2+frame_width-button_size] = res
    
    cv2.putText(frame, f'{c}won payed ok!', (frame_width//2-200,frame_height//2),font,1.3,(255,255,255),lineType)
    menus[n] = []
cnt = 0
cap = cv2.VideoCapture(0)
with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5,min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("카메라를 찾을 수 없습니다.")
            continue
        frame = cv2.flip(frame,1)
        
        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)
        frame_height, frame_width, _ = frame.shape
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        hand_x = 0
        hand_y = 0
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())           
                hand_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x*frame_width
                hand_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y*frame_height
            frame = draw_keyboard(frame)
            calculating_hand_org()
            
            if click_state == 1:
                typing_calculate(hand_x,hand_y)
        if state:
            if cnt >= 100:
                state = False
                for idx, p in enumerate(payment):
                    if p == c_pay:
                        payment[idx] = '0'
                
            else:
                done(c_pay,n)
        else:
            cnt = 0
            
        cv2.imshow('frame', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break
        cnt+=1
cap.release()
cv2.destroyAllWindows()