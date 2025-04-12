import cv2
import mediapipe as mp
import numpy as np

# Inicialización de MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Variables para el objeto virtual
class DraggableRectangle:
    def __init__(self, pos, size=(100, 100)):
        self.pos = pos
        self.size = size
        self.dragging = False

    def update(self, cursor):
        cx, cy = self.pos
        w, h = self.size

        if self.dragging:
            self.pos = (cursor[0] - w // 2, cursor[1] - h // 2)

    def is_cursor_inside(self, cursor):
        cx, cy = self.pos
        w, h = self.size
        return cx < cursor[0] < cx + w and cy < cursor[1] < cy + h

    def draw(self, img):
        cx, cy = self.pos
        w, h = self.size
        color = (0, 255, 0) if self.dragging else (255, 0, 255)
        cv2.rectangle(img, (cx, cy), (cx + w, cy + h), color, cv2.FILLED)
        return img

# Crear un rectángulo
rect = DraggableRectangle((250, 250))

# Captura de video
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Conversión a RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    cursor = (0, 0)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            # Dibujar puntos
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Coordenadas de pulgar (4) e índice (8)
            x1, y1 = lm_list[4]
            x2, y2 = lm_list[8]

            # Centro entre los dedos
            cursor = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.circle(frame, cursor, 15, (255, 255, 255), cv2.FILLED)

            # Distancia entre dedos
            distance = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))

            # Detectar agarre (dedos juntos)
            if distance < 40:
                if rect.is_cursor_inside(cursor):
                    rect.dragging = True
            else:
                rect.dragging = False

            rect.update(cursor)

    rect.draw(frame)
    cv2.imshow("Mini proyecto inteligencia artificial ", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
