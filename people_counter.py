import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

line_y = 300
count = 0
crossed_ids = set()

while True:

    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, tracker="bytetrack.yaml")

    boxes = results[0].boxes

    if boxes.id is not None:

        for box, track_id, cls in zip(boxes.xyxy, boxes.id, boxes.cls):

            # sadece insanları say
            if int(cls) != 0:
                continue

            x1,y1,x2,y2 = map(int,box)
            track_id = int(track_id)

            cx = int((x1+x2)/2)
            cy = int((y1+y2)/2)

            if cy > line_y and track_id not in crossed_ids:

                crossed_ids.add(track_id)
                count += 1

            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)

            cv2.putText(frame,f"ID {track_id}",(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)

    cv2.line(frame,(0,line_y),(640,line_y),(0,255,0),3)

    cv2.putText(frame,f"Count: {count}",(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

    cv2.imshow("People Counter",frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()