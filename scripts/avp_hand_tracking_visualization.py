import sys
import time

import numpy as np
import rerun as rr
from avp_teleop import VisionProStreamer
from scipy.spatial.transform import Rotation as R


def log_transform_axes(name: str, transform: np.ndarray, scale: float = 0.1) -> None:
    """
    Given a 4x4 transformation matrix, log a 3D axis at the joint.
    - `name` is the channel name (e.g., "head" or "right_fingers/3").
    - `scale` determines the length of the drawn axes.
    """
    # Extract the origin and the three local axes:
    origin = transform[:3, 3]
    x_dir = transform[:3, 0]
    y_dir = transform[:3, 1]
    z_dir = transform[:3, 2]

    # Compute endpoints for the axes lines:
    x_end = origin + scale * x_dir
    y_end = origin + scale * y_dir
    z_end = origin + scale * z_dir

    # Log the x-axis (red), y-axis (green), and z-axis (blue).
    # (We assume that rr.LineStrip3D takes an array of positions and an array of colors.)
    rr.log(
        f"{name}/axis/x",
        rr.LineStrips3D(np.array([origin, x_end]), colors=np.array([[255, 0, 0]])),  # red for x
    )
    rr.log(
        f"{name}/axis/y",
        rr.LineStrips3D(np.array([origin, y_end]), colors=np.array([[0, 255, 0]])),  # green for y
    )
    rr.log(
        f"{name}/axis/z",
        rr.LineStrips3D(np.array([origin, z_end]), colors=np.array([[0, 0, 255]])),  # blue for z
    )


def main():

    # The 'spawn=True' argument opens the GUI viewer in a separate window.
    rr.init("hand_tracking", spawn=True)

    # Connect to the AVP device streaming hand tracking data.
    # avp_ip = "192.168.0.111"
    avp_ip = "192.168.1.163"
    streamer = VisionProStreamer(ip=avp_ip, record=False)
    streamer.start_streaming()

    start = time.time()

    while True:
        # Get the latest data from the stream
        data = streamer.latest

        # Process head data (shape: (1,4,4))
        head_transform = data["head"][0]
        
        rotation = R.from_euler("z", [np.pi / 2])
        matrix = np.eye(4)
        matrix[:3, :3] = rotation.as_matrix()
        head_transform = np.dot(matrix, head_transform)
        
        log_transform_axes("head", head_transform)

        # Process right and left wrist data (each shape: (1,4,4))
        # rotate around the x axis by 90 degrees to align with the x axis

        right_wrist_transform = data["right_wrist"][0]
        left_wrist_transform = data["left_wrist"][0]

        # # compute the relative transform from the left wrist to the head
        # left_wrist_to_head = np.dot(np.linalg.inv(left_wrist_transform), head_transform)
        # print(np.linalg.inv(left_wrist_to_head))
        
        # compute the relative transform from the right wrist to the head
        right_wrist_to_head = np.dot(np.linalg.inv(right_wrist_transform), head_transform)
        print(np.linalg.inv(right_wrist_to_head))
        
        log_transform_axes("right_wrist", right_wrist_transform)
        log_transform_axes("left_wrist", left_wrist_transform)

        # Process finger joints.
        # Each finger joint array is of shape (25,4,4).
        right_fingers = data["right_fingers"]
        left_fingers = data["left_fingers"]

        for i in range(right_fingers.shape[0]):
            finger_transform = right_fingers[i]
            finger_transform = np.dot(right_wrist_transform, finger_transform)
            log_transform_axes(f"right_fingers/{i}", finger_transform, scale=0.02)

        for i in range(left_fingers.shape[0]):
            finger_transform = left_fingers[i]
            finger_transform = np.dot(left_wrist_transform, finger_transform)
            log_transform_axes(f"left_fingers/{i}", finger_transform, scale=0.02)

        # Optionally, print extra scalar data to the console.
        # print("Right pinch distance:", data['right_pinch_distance'])
        # print("Left pinch distance:", data['left_pinch_distance'])
        # print("Right wrist roll:", data['right_wrist_roll'])
        # print("Left wrist roll:", data['left_wrist_roll'])

        # Small sleep to roughly target a 30 Hz update rate.
        time.sleep(1 / 33)
        
        if time.time() - start > 5:
            break

    rr.save("hand_tracking")
    streamer.stop_streaming()


if __name__ == "__main__":
    main()
