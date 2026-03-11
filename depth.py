import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import cv2
import torch

# model yükle
model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
model.eval()

# transform
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    input_batch = transform(img)

    with torch.no_grad():
        prediction = model(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()

    # normalize depth
    depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth = depth.astype("uint8")

    # görüntünün ortası
    h, w = depth.shape
    center_depth = depth[h//2, w//2]

    # depth değerini yaz
    cv2.putText(frame,
                f"Depth value: {center_depth}",
                (40,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2)

    # ortayı işaretle
    cv2.circle(frame,(w//2,h//2),5,(0,0,255),-1)

    cv2.imshow("Camera", frame)
    cv2.imshow("Depth Map", depth)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()