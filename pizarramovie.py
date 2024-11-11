import cv2
import mediapipe as mp
import numpy as np

# Inicializar MediaPipe Hand
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Colores 
colors = {
    "green": (0, 255, 0),        
    "yellow": (0, 255, 255),     
    "blue": (255, 0, 0),         
    "red": (0, 0, 255)          
}

# Posiciones de los colores en la pantalla
color_positions = {
    "green": (50, 50),
    "yellow": (150, 50),
    "blue": (250, 50),
    "red": (350, 50)
}


cap = cv2.VideoCapture(0)


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

#inicializa con verde el dedo
current_color = colors["green"]


drawing_frame = None
is_eraser_mode = False  

while True:
    
    ret, frame = cap.read()
    
    if not ret:
        break
    
   
    #frame = cv2.resize(frame, (640, 480))  

    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
   
    results = hands.process(frame_rgb)
    
   
    for color, position in color_positions.items():
        cv2.rectangle(frame, position, (position[0] + 50, position[1] + 50), colors[color], -1)
        cv2.putText(frame, color.capitalize(), (position[0] + 10, position[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            
            
            finger_position = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            finger_x = int(finger_position.x * frame.shape[1])
            finger_y = int(finger_position.y * frame.shape[0])
            
           
            thumb_position = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_x = int(thumb_position.x * frame.shape[1])
            thumb_y = int(thumb_position.y * frame.shape[0])
            
            
            middle_finger_position = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            middle_finger_x = int(middle_finger_position.x * frame.shape[1])
            middle_finger_y = int(middle_finger_position.y * frame.shape[0])
            
            
            ring_finger_position = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            ring_finger_x = int(ring_finger_position.x * frame.shape[1])
            ring_finger_y = int(ring_finger_position.y * frame.shape[0])
            
            if abs(finger_x - thumb_x) < 50 and abs(finger_y - thumb_y) < 50:
                if drawing_frame is None:
                    drawing_frame = np.zeros_like(frame)  # Crear un frame vacío para el dibujo

                # Si el dedo índice está extendido y tocando, dibujar
                cv2.circle(drawing_frame, (finger_x, finger_y), 10, current_color, -1)

            # Si el dedo índice y el dedo medio están lo suficientemente cerca, activar el borrador
            if abs(finger_x - middle_finger_x) < 50 and abs(finger_y - middle_finger_y) < 50:
                is_eraser_mode = True  # Activar el borrador

            # Si el dedo índice y el dedo medio no están cerca, desactivar el borrador
            if not (abs(finger_x - middle_finger_x) < 50 and abs(finger_y - middle_finger_y) < 50):
                is_eraser_mode = False  # Desactivar el borrador

            #  borrar las partes donde pasa el dedo
            if is_eraser_mode:
                
                cv2.circle(drawing_frame, (finger_x, finger_y), 20, (0, 0, 0), -1)

            # Si el dedo anular y el pulgar están lo suficientemente cerca, borrar todo lo dibujado
            if abs(thumb_x - ring_finger_x) < 50 and abs(thumb_y - ring_finger_y) < 50:
                drawing_frame = np.zeros_like(frame)  

            # Si el dedo índice toca uno de los colores, cambiar el color
            for color, position in color_positions.items():
                color_x, color_y = position
                if color_x < finger_x < color_x + 50 and color_y < finger_y < color_y + 50:
                    current_color = colors[color]

    # Mostrar el resultado con el dibujo sobre la cámara normal
    if drawing_frame is not None:
        final_frame = cv2.addWeighted(frame, 1, drawing_frame, 0.7, 0)  
    else:
        final_frame = frame

    cv2.imshow("Interactive Whiteboard", final_frame)
    
    # Salir cuando se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
