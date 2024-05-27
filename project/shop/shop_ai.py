import cv2
import json
from ultralytics import YOLO


class ShopAI:
    # Init
    def __init__(self):
        # VideoCapture
        self.capture = cv2.VideoCapture(0)
        self.capture.set(3, 1280)
        self.capture.set(4, 720)

        # MODELS:
        # Object model
        self.ObjectModel = YOLO("models/yolov8l.onnx", task="detect")
        # Bancopoly model
        self.BancopolyModel = YOLO("models/bancopoly.onnx", task="detect")

        self.clsObject = [
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]

        self.clsBancopoly = ["billete_10", "billete_100", "billete_20", "billete_50"]

    def get_name_from_database(self, object_name):
        with open("db/Productos.json") as file:
            data = json.load(file)

        if object_name in data:
            return data[object_name]["name"]
        else:
            return "Desconocido"

    def get_price_from_database(self, object_name):
        with open("db/Productos.json") as file:
            data = json.load(file)

        if object_name in data:
            return float(data[object_name]["price"])
        else:
            return 0

    def start(self):
        cv2.namedWindow("Camera Capture", cv2.WINDOW_NORMAL)

        while True:
            # Capture frame-by-frame
            ret, frame = self.capture.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Object Detection
            results_object = self.ObjectModel(frame, stream=True, verbose=False)
            results_bancopoly = self.BancopolyModel(
                frame, imgsz=640, conf=0.5, stream=True, verbose=False
            )

            # Variables to store the total price and the list of detected products
            total_price = 0
            detected_products = []

            # Delay for smoother display
            wait_key_delay = 5

            # Text thickness and font scale
            thickness = 1
            font_scale = 0.5

            # Draw bounding boxes around detected objects
            for res in results_object:
                boxes = res.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    cls = int(box.cls[0])
                    print(f"Detected object class: {cls} ({self.clsObject[cls]})")

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        thickness,
                    )

                    product_name = self.get_name_from_database(self.clsObject[cls])
                    price = self.get_price_from_database(self.clsObject[cls])

                    formatted_price = "Q. {:,.2f}".format(price)

                    cv2.putText(
                        frame,
                        f"{product_name} {formatted_price}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (0, 255, 0),
                        thickness,
                    )

                    total_price += price
                    detected_products.append(f"{product_name} Q. {formatted_price}")

            for res in results_bancopoly:
                boxes = res.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    cls = int(box.cls[0])

                    if cls < len(self.clsBancopoly):
                        print(
                            f"Detected Bancopoly class: {cls} ({self.clsBancopoly[cls]})"
                        )

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (255, 0, 0),
                            thickness,
                        )

                        bancopoly_name = self.clsBancopoly[cls]
                        bancopoly_value = int(bancopoly_name.split("_")[1])

                        formatted_value = "Q. {:,.2f}".format(bancopoly_value)

                        cv2.putText(
                            frame,
                            f"{bancopoly_name} {formatted_value}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            (255, 0, 0),
                            thickness,
                        )

                        total_price -= bancopoly_value
                        detected_products.append(
                            f"{bancopoly_name} Q. {formatted_value}"
                        )
                    else:
                        print(
                            f"Warning: Detected class index {cls} is out of range for Bancopoly classes"
                        )

            left_margin = 0
            line_height = 25
            formatted_total = "Q. {:,.2f}".format(total_price)
            cv2.putText(
                frame,
                f"Total: {formatted_total}",
                (left_margin, line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 255, 0),
                thickness,
            )

            for i, product in enumerate(detected_products):
                cv2.putText(
                    frame,
                    f"{i+1}. {product}",
                    (left_margin, (i + 2) * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 255, 0),
                    thickness,
                )

            # Display the frame
            cv2.imshow("Camera Capture", frame)

            # Check for key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Delay for smoother display
            cv2.waitKey(wait_key_delay)

        self.capture.release()
        cv2.destroyAllWindows()
        return
