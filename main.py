from ultralytics import YOLO
import cv2
import numpy as np

# nodo para arbol AVL
class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.height = 1

# arbol AVL para almacenar velocidades
class AVLTree:
    def __init__(self):
        self.root = None

    # obtener altura del nodo
    def get_height(self, node):
        return node.height if node else 0

    # calcular el balance del nodo
    def get_balance(self, node):
        return self.get_height(node.left) - self.get_height(node.right) if node else 0

    # rotacion derecha para balancear
    def right_rotate(self, z):
        y = z.left
        if not y:
            return z
        T3 = y.right
        y.right = z
        z.left = T3
        z.height = 1 + max(self.get_height(z.left), self.get_height(z.right))
        y.height = 1 + max(self.get_height(y.left), self.get_height(y.right))
        return y

    # rotacion izquierda para balancear
    def left_rotate(self, z):
        y = z.right
        if not y:
            return z
        T2 = y.left
        y.left = z
        z.right = T2
        z.height = 1 + max(self.get_height(z.left), self.get_height(z.right))
        y.height = 1 + max(self.get_height(y.left), self.get_height(y.right))
        return y

    # insertar una nueva velocidad en el arbol
    def insert(self, root, key):
        if not root:
            return Node(key)
        if key < root.key:
            root.left = self.insert(root.left, key)
        else:
            root.right = self.insert(root.right, key)

        root.height = 1 + max(self.get_height(root.left), self.get_height(root.right))
        balance = self.get_balance(root)

        # balancear el arbol si es necesario
        if balance > 1 and key < root.left.key:
            return self.right_rotate(root)
        if balance < -1 and key > root.right.key:
            return self.left_rotate(root)
        if balance > 1 and key > root.left.key:
            root.left = self.left_rotate(root.left)
            return self.right_rotate(root)
        if balance < -1 and key < root.right.key:
            root.right = self.right_rotate(root.right)
            return self.left_rotate(root)
        return root

    # funcion publica para insertar
    def add_speed(self, speed):
        self.root = self.insert(self.root, speed)

# calcula velocidad en km/h entre dos puntos
def calc_speed_kmh(p1, p2, time_sec, mpp):
    dist_pix = np.linalg.norm(np.array(p2) - np.array(p1))
    dist_m = dist_pix * mpp
    speed_m_s = dist_m / time_sec
    raw_speed = speed_m_s * 3.6
    corrected_speed = raw_speed * 0.9
    return corrected_speed

# estimar metros por pixel segun altura del objeto
def estimate_mpp(cy, frame_height):
    return 0.01 + (cy / frame_height) * 0.03

# analiza lista de velocidades y da una recomendacion
def analyze_speeds_v2(speeds, speed_limit, street_type):
    exceed_speeds = [s for s in speeds if s > speed_limit + 5]
    if not exceed_speeds:
        return "velocidad dentro de limites aceptables. no se requiere accion."

    avg_exceed_speed = sum(exceed_speeds) / len(exceed_speeds)
    diff = avg_exceed_speed - speed_limit

    if street_type == "autopista":
        if diff > 25:
            return "velocidad dentro de limites aceptables. no se requiere accion."
        else:
            return "velocidad levemente superior al limite; considerar senaletica."

    if street_type == "avenida":
        if diff > 15:
            return "velocidad elevada: se recomienda colocar senaletica o reductores."
        else:
            return "velocidad dentro de limites aceptables. no se requiere accion."

    if street_type == "calle":
        if diff > 5:
            return "exceso grave: se recomienda instalar topes para reducir velocidad."
        else:
            return "velocidad levemente superior al limite; vigilar y considerar medidas leves."

# funcion principal
def main():
    model = YOLO('yolov8n.pt')  # carga modelo de deteccion
    cap = cv2.VideoCapture('road.mp4')  # carga video

    # pide al usuario datos iniciales
    street_type = input("ingrese el tipo de calle (autopista/calle/avenida): ").strip().lower()
    speed_limit = float(input("ingrese el limite de velocidad en km/h: "))

    prev_positions = {}
    frame_rate = cap.get(cv2.CAP_PROP_FPS) or 30

    ret, frame = cap.read()
    if not ret:
        print("no se pudo leer el video.")
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    speed_tree = AVLTree()
    speeds_list = []
    car_speeds = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]  # deteccion
        current_positions = {}

        for box, cls_id in zip(results.boxes.xyxy, results.boxes.cls):
            if int(cls_id) in [2, 5, 7]:  # solo coches, buses y camiones
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # buscar coincidencia con objetos del frame anterior
                min_id = None
                min_dist = 1e9
                for id_, pos in prev_positions.items():
                    dist = np.linalg.norm(np.array([cx, cy]) - np.array(pos))
                    if dist < min_dist and dist < 50:
                        min_dist = dist
                        min_id = id_
                if min_id is None:
                    min_id = len(prev_positions) + len(current_positions) + 1
                current_positions[min_id] = (cx, cy)

                # calcular velocidad si hay posicion anterior
                speed_kmh = 0
                if min_id in prev_positions:
                    mpp = estimate_mpp(cy, frame.shape[0])
                    speed_kmh = calc_speed_kmh(prev_positions[min_id], (cx, cy), 1/frame_rate, mpp)

                # guardar velocidad por vehiculo
                if min_id not in car_speeds:
                    car_speeds[min_id] = []
                car_speeds[min_id].append(speed_kmh)

                # color segun exceso de velocidad
                if speed_kmh > speed_limit + 25:
                    color = (0, 0, 255)
                elif speed_kmh > speed_limit + 5:
                    color = (0, 255, 255)
                else:
                    color = (0, 255, 0)

                # dibujar caja y velocidad
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{speed_kmh:.1f} km/h', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        prev_positions = current_positions.copy()
        cv2.imshow("velocidad vehiculos", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # calcular promedios corregidos por vehiculo
    for speeds in car_speeds.values():
        if speeds:
            avg = sum(speeds) / len(speeds)
            corrected_avg = max(avg - 15, 0)
            speeds_list.append(corrected_avg)
            speed_tree.add_speed(corrected_avg)

    # mostrar recomendacion final
    recommendation = analyze_speeds_v2(speeds_list, speed_limit, street_type)
    print(f"\n--- recomendacion final para la {street_type} ---")
    print(recommendation)

if __name__ == "__main__":
    main()
