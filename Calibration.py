import cv2, os, json, numpy
import Mapping, Tuning, MappingFile, WindowUtils
from typing import Tuple

FRAME = numpy.ndarray


# CHOOSE, CROP, AND PREPARE FRAME
def prepare_video_mapping(cap: cv2.VideoCapture, global_rotation: int, progress_str: str = "") -> Tuple[
    int, FRAME, dict]:
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, first_frame = cap.read()
    if not ret:
        return 0, None, None

    (orig_h, orig_w) = first_frame.shape[:2]
    center = (orig_w // 2, orig_h // 2)

    win_scrub = f"{progress_str} Step 2A: Scrub to find a good frame (Press ENTER)".strip()
    cv2.namedWindow(win_scrub)
    WindowUtils.apply_brief_topmost(win_scrub)

    current_data = [0, first_frame.copy()]

    def on_trackbar(val):
        cap.set(cv2.CAP_PROP_POS_FRAMES, val)
        ret, frame = cap.read()
        if ret:
            current_data[0] = val
            current_data[1] = frame

            if global_rotation % 360 != 0:
                M = cv2.getRotationMatrix2D(center, global_rotation, 1.0)
                disp_frame = cv2.warpAffine(frame, M, (orig_w, orig_h))
            else:
                disp_frame = frame.copy()

            cv2.putText(disp_frame, f"Frame: {val}/{total_frames - 1} | Global Rot: {-global_rotation}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(win_scrub, disp_frame)

    cv2.createTrackbar("Frame", win_scrub, 0, total_frames - 1, on_trackbar)
    on_trackbar(0)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32):
            break
        elif key == 27:
            cv2.destroyWindow(win_scrub)
            exit(1)
        WindowUtils.enforce_window_alive(win_scrub)

    cv2.destroyWindow(win_scrub)

    frame_idx = current_data[0]
    raw_frame = current_data[1]

    if global_rotation % 360 != 0:
        M = cv2.getRotationMatrix2D(center, global_rotation, 1.0)
        rotated_frame = cv2.warpAffine(raw_frame, M, (orig_w, orig_h))
    else:
        rotated_frame = raw_frame.copy()

    win_crop = f"{progress_str} Step 2B: Crop (Drag tight box around digits, then press ENTER)".strip()
    WindowUtils.show_centered_window(win_crop, rotated_frame)
    roi = cv2.selectROI(win_crop, rotated_frame, showCrosshair=True, fromCenter=False, printNotice=False)
    cv2.destroyWindow(win_crop)

    x, y, w, h = roi
    if w == 0 or h == 0:
        x, y, w, h = 0, 0, orig_w, orig_h

    preproc_data = {
        "rotation_angle": global_rotation,
        "center": center,
        "roi": [x, y, w, h]
    }

    cropped_frame = rotated_frame[y:y + h, x:x + w]
    calib_frame = Tuning.clean_frame(cropped_frame, return_bgr=True, preview_grayscale=False)

    return frame_idx, calib_frame, preproc_data


def calibrate_video(video_path, auto_mode: bool = True, radius: int = 2, progress_str: str = "") -> Tuple[
    'COORDINATES_MAP', dict]:
    print(f'Calibrating {video_path}')

    config = Tuning.get_config()
    global_rotation = config.get("global_rotation", 0)

    with WindowUtils.VideoCapture(video_path) as cap:
        frame_number, frame, preproc_data = prepare_video_mapping(cap, global_rotation, progress_str)

    if frame is None:
        return None, None

    coordinates_map = Mapping.extract_coordinates_map(frame_number, frame, auto_mode, radius, progress_str)
    print(coordinates_map)

    return coordinates_map, preproc_data


def main(auto_mode: bool = True, radius: int = 2):
    if not os.path.exists('config.json'):
        print("Notice: 'config.json' not found. Please run Step 1 (Tuning) first to set Master Configuration.")

    video_paths = WindowUtils.open_file_dialog(title="Select Raw Videos to Crop & Map",
                                               filetypes=[("MP4 files", "*.mp4")], multiple=True)

    if not video_paths:
        return

    total_videos = len(video_paths)

    for idx, video_path in enumerate(video_paths):
        video_folder = os.path.dirname(video_path)
        video_filename = os.path.basename(video_path)
        base_name = os.path.splitext(video_filename)[0]
        maps_folder = os.path.join(video_folder, 'maps')

        progress_str = f"[{idx + 1}/{total_videos}]"

        coordinates_map, preproc_data = calibrate_video(video_path, auto_mode, radius, progress_str)

        if coordinates_map and preproc_data:
            MappingFile.save_coordinates_map(coordinates_map, maps_folder, video_filename)

            preproc_folder = os.path.join(video_folder, 'preprocessing')
            os.makedirs(preproc_folder, exist_ok=True)

            preproc_path = os.path.normpath(os.path.join(preproc_folder, f"{base_name}.json"))
            with open(preproc_path, 'w') as f:
                json.dump(preproc_data, f, indent=4)
            print(f"Saved preprocessing crop to {preproc_path}")


if __name__ == '__main__':
    main()