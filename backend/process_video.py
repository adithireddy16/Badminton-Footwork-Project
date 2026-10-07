import cv2
import json
import numpy as np
from ultralytics import YOLO


# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = "runs/detect/train-3/weights/best.pt"
COURT_POINTS_PATH = "court_points.json"

CONFIDENCE = 0.20
COURT_TOLERANCE = 50
MAX_PLAYERS = 2


# =========================================================
# LOAD YOLO MODEL
# =========================================================

model = YOLO(MODEL_PATH)

print("YOLOv8 model loaded in process_video.py")


# =========================================================
# LOAD COURT POINTS
# =========================================================

def load_court_points():

    with open(COURT_POINTS_PATH, "r") as file:
        data = json.load(file)

    # Handle different JSON formats
    if isinstance(data, list):
        points = data

    elif "points" in data:
        points = data["points"]

    elif "court_points" in data:
        points = data["court_points"]

    else:
        raise ValueError("Court points not found in court_points.json")

    return np.array(points, dtype=np.float32)


court_points = load_court_points()

print("Court points loaded successfully!")


# =========================================================
# PERSPECTIVE TRANSFORM
# =========================================================

destination_points = np.array(
    [
        [0, 0],
        [1000, 0],
        [1000, 1000],
        [0, 1000]
    ],
    dtype=np.float32
)

perspective_matrix = cv2.getPerspectiveTransform(
    court_points,
    destination_points
)


# =========================================================
# CHECK WHETHER PLAYER IS INSIDE COURT
# =========================================================

def is_inside_court(x, y):

    polygon = court_points.astype(np.int32)

    distance = cv2.pointPolygonTest(
        polygon,
        (float(x), float(y)),
        True
    )

    return distance >= -COURT_TOLERANCE


# =========================================================
# SIX-CORNER CLASSIFICATION
# =========================================================

def classify_position(x, y):

    # Left side = Backhand
    # Right side = Forehand

    if x < 500:
        stroke = "Backhand"
    else:
        stroke = "Forehand"

    # Court depth
    if y < 333:
        area = "Back-Court"

    elif y < 666:
        area = "Side/Mid"

    else:
        area = "Front-Court"

    return stroke + " " + area


# =========================================================
# PROCESS VIDEO
# =========================================================

def process_video(input_video, output_video):

    print("Starting six-corner video processing...")

    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():

        print("❌ Could not open input video.")

        return False


    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )


    print("Width:", width)
    print("Height:", height)
    print("FPS:", fps)


    # -----------------------------------------------------
    # OUTPUT VIDEO
    # -----------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    out = cv2.VideoWriter(
        output_video,
        fourcc,
        fps,
        (width, height)
    )


    frame_number = 0


    # =====================================================
    # FRAME LOOP
    # =====================================================

    while True:

        success, frame = cap.read()

        if not success:
            break


        # -------------------------------------------------
        # YOLO DETECTION
        # -------------------------------------------------

        results = model(
            frame,
            conf=CONFIDENCE,
            verbose=False
        )


        players = []


        # -------------------------------------------------
        # GET DETECTIONS
        # -------------------------------------------------

        for result in results:

            if result.boxes is None:
                continue


            for box in result.boxes:

                coordinates = box.xyxy[0].cpu().numpy()

                x1, y1, x2, y2 = map(
                    int,
                    coordinates
                )


                confidence = float(
                    box.conf[0].cpu().numpy()
                )


                # Bottom-center of bounding box
                foot_x = int(
                    (x1 + x2) / 2
                )

                foot_y = int(y2)


                # Check court
                if not is_inside_court(
                    foot_x,
                    foot_y
                ):
                    continue


                players.append(
                    {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "foot_x": foot_x,
                        "foot_y": foot_y,
                        "confidence": confidence
                    }
                )


        # -------------------------------------------------
        # KEEP ONLY TWO PLAYERS
        # -------------------------------------------------

        players = sorted(
            players,
            key=lambda p: p["confidence"],
            reverse=True
        )

        players = players[:MAX_PLAYERS]


        # -------------------------------------------------
        # DRAW PLAYER INFORMATION
        # -------------------------------------------------

        for index, player in enumerate(players):

            x1 = player["x1"]
            y1 = player["y1"]
            x2 = player["x2"]
            y2 = player["y2"]

            foot_x = player["foot_x"]
            foot_y = player["foot_y"]

            confidence = player["confidence"]


            # Transform foot point to court coordinates
            point = np.array(
                [
                    [
                        [
                            foot_x,
                            foot_y
                        ]
                    ]
                ],
                dtype=np.float32
            )


            transformed = cv2.perspectiveTransform(
                point,
                perspective_matrix
            )


            court_x = float(
                transformed[0][0][0]
            )

            court_y = float(
                transformed[0][0][1]
            )


            # Six-corner classification
            label = classify_position(
                court_x,
                court_y
            )


            # Player name
            if index == 0:
                player_name = "PLAYER A"
            else:
                player_name = "PLAYER B"


            # Percentage
            percentage = int(
                confidence * 100
            )


            # -------------------------------------------------
            # DRAW BOX
            # -------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # -------------------------------------------------
            # DRAW PLAYER NAME
            # -------------------------------------------------

            cv2.putText(
                frame,
                player_name,
                (
                    x1,
                    max(y1 - 35, 25)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )


            # -------------------------------------------------
            # DRAW MOVEMENT LABEL
            # -------------------------------------------------

            label_text = (
                f"{label.upper()} {percentage}%"
            )


            cv2.putText(
                frame,
                label_text,
                (
                    x1,
                    max(y1 - 10, 50)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )


            # -------------------------------------------------
            # DRAW FOOT POINT
            # -------------------------------------------------

            cv2.circle(
                frame,
                (foot_x, foot_y),
                5,
                (0, 0, 255),
                -1
            )


        # -------------------------------------------------
        # WRITE FRAME
        # -------------------------------------------------

        out.write(frame)

        frame_number += 1


        if frame_number % 100 == 0:

            print(
                "Processed frames:",
                frame_number
            )


    # =====================================================
    # FINISH
    # =====================================================

    cap.release()
    out.release()


    print("✅ Six-corner video processing completed!")

    return True