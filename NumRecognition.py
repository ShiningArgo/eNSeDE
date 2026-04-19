import cv2, csv, os, json, multiprocessing, copy
import numpy as np
from datetime import datetime
from typing import Tuple, List
import MappingFile, Tuning, WindowUtils

RESULT = Tuple

DIGITS_SEGMENTS = {}
DIGITS_SEGMENTS_COORDINATES = None

# --- MULTIPROCESSING SHARED STATE ---
SHARED_PROGRESS_ARRAY = None


def init_worker(progress_arr):
    """Initializes the shared memory array for each worker process."""
    global SHARED_PROGRESS_ARRAY
    SHARED_PROGRESS_ARRAY = progress_arr


# LOAD SEGMENTS DEFINITION
def load_segments_definition():
    global DIGITS_SEGMENTS
    map_file = 'segments_map.json'
    if not os.path.exists(map_file):
        map_file = 'segments_map_default.json'
        if not os.path.exists(map_file):
            print("Error: Neither segments_map.json nor segments_map_default.json found.")
            exit(1)

    with open(map_file, 'r') as f:
        data = json.load(f)

    DIGITS_SEGMENTS = {}
    for k, v in data['mappings'].items():
        try:
            # Parse the new binary string format back into base-10 integers
            key_int = int(k, 2)
        except ValueError:
            # Fallback for older config files that still use base-10 string keys
            key_int = int(k)

        DIGITS_SEGMENTS[key_int] = v

    print(f"Loaded {len(DIGITS_SEGMENTS)} segment definitions from {map_file} (N={data['segments_count']})")


# APPLY PREPROCESSING
def apply_preprocessing(frame, preproc_data):
    if not preproc_data:
        return frame
    angle = preproc_data.get("rotation_angle", 0)
    x, y, w, h = preproc_data["roi"]
    if angle % 360 != 0:
        center = tuple(preproc_data["center"])
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        M[0, 2] -= x
        M[1, 2] -= y
        return cv2.warpAffine(frame, M, (w, h))
    else:
        return frame[y:y + h, x:x + w]


# EVALUATE SEGMENTS
def evaluate_segments(frame, coords_map, radius, weights):
    digits_data = []
    for segments in coords_map:
        seg_data = []
        for segment_pos, (cx, cy) in enumerate(segments):
            y_start = max(0, cy - radius)
            y_end = min(frame.shape[0], cy + radius + 1)
            x_start = max(0, cx - radius)
            x_end = min(frame.shape[1], cx + radius + 1)

            w_y_start = radius - (cy - y_start)
            w_y_end = radius + (y_end - cy)
            w_x_start = radius - (cx - x_start)
            w_x_end = radius + (x_end - cx)

            roi = frame[y_start:y_end, x_start:x_end]
            active_weights = weights[w_y_start:w_y_end, w_x_start:w_x_end]

            if roi.size == 0 or np.sum(active_weights) == 0:
                seg_data.append({'pos': segment_pos, 'score': 0.0, 'is_on': False, 'conf': 1.0})
                continue

            black_pixels_weight = np.sum((roi == 0) * active_weights)
            total_active_weight = np.sum(active_weights)

            score = black_pixels_weight / total_active_weight
            is_black = score > 0.5
            conf = score if is_black else 1.0 - score

            seg_data.append({'pos': segment_pos, 'score': score, 'is_on': is_black, 'conf': conf})
        digits_data.append(seg_data)
    return digits_data


# EXTRACT STATE RESULTS
def extract_state_results(digits_data, seg_map):
    result = ''
    total_conf = 0.0
    num_digits = len(digits_data)

    for seg_data in reversed(digits_data):
        binary_res = 0
        digit_conf = 0.0
        for seg in seg_data:
            if seg['is_on']:
                binary_res |= 1 << seg['pos']
            digit_conf += seg['conf']

        avg_conf = (digit_conf / len(seg_data)) * 100 if seg_data else 0
        char = seg_map.get(binary_res, '?')
        result += str(char)
        total_conf += avg_conf

    return result, round((total_conf / num_digits) if num_digits else 0, 2)


# PROCESS OCR - Outputs Raw State and Best Valid Dictionary Match
def process_ocr(digits_data, seg_map):
    base_char, base_conf = extract_state_results(digits_data, seg_map)

    best_valid_str = ''
    total_best_conf = 0.0
    num_digits = len(digits_data)

    for seg_data in reversed(digits_data):
        best_char = '?'
        best_conf = -1.0

        for bin_key, char in seg_map.items():
            key_conf = 0.0
            for seg in seg_data:
                requires_on = bool(bin_key & (1 << seg['pos']))
                seg_c = seg['score'] if requires_on else 1.0 - seg['score']
                key_conf += seg_c

            avg_key_conf = (key_conf / len(seg_data)) * 100 if seg_data else 0
            if avg_key_conf > best_conf:
                best_conf = avg_key_conf
                best_char = char

        best_valid_str += str(best_char)
        total_best_conf += best_conf

    best_valid_conf = round((total_best_conf / num_digits) if num_digits else 0, 2)

    return base_char, base_conf, best_valid_str, best_valid_conf


# WORKER OCR
def worker_ocr(args):
    global SHARED_PROGRESS_ARRAY
    video_idx, video_path, start_frame, end_frame, fps, coords_map, seg_map, radius, preproc_data, process_every_n, global_offset = args
    results = []

    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    distance = np.sqrt(x ** 2 + y ** 2)
    weights = np.clip(1.0 - (distance / (radius + 1.0)), 0.0, 1.0)

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for frame_num in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret: break

        if frame_num % process_every_n != 0:
            if SHARED_PROGRESS_ARRAY is not None:
                SHARED_PROGRESS_ARRAY[global_offset + frame_num] = 2
            continue

        frame = apply_preprocessing(frame, preproc_data)
        cleaned_frame = Tuning.clean_frame(frame)
        digits_data = evaluate_segments(cleaned_frame, coords_map, radius, weights)

        base_char, base_conf, best_char, best_conf = process_ocr(digits_data, seg_map)
        results.append((frame_num, frame_num / fps, base_char, base_conf, best_char, best_conf))

        if SHARED_PROGRESS_ARRAY is not None:
            SHARED_PROGRESS_ARRAY[global_offset + frame_num] = 1

    cap.release()
    return video_idx, results


# BUILD DYNAMIC CSV HEADERS
def get_csv_headers(merged: bool):
    headers = []
    if merged:
        headers.append('Video_Name')

    headers.extend(['Frame', 'Timestamp (s)', 'Detected_Number', 'Confidence (%)', 'Best_Valid', 'Best_Valid_Conf (%)'])
    return headers


def save_to_csv(output_folder: str, video_name: str, results: List[RESULT]):
    base_name = os.path.splitext(video_name)[0]
    output_csv = os.path.normpath(os.path.join(output_folder, f"{base_name}.csv"))
    headers = get_csv_headers(merged=False)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(results)
    print(f"Saved: {output_csv}")


def save_merged_csv(output_folder: str, filename: str, results: List[Tuple]):
    output_csv = os.path.normpath(os.path.join(output_folder, f"{filename}.csv"))
    headers = get_csv_headers(merged=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(results)
    print(f"\nSaved merged results to: {output_csv}")


# PROCESS MULTIPLE VIDEOS CONCURRENTLY
def process_multiple_videos(videos_info: List[dict], radius: int, process_every_n: int) -> dict:
    total_frames_all = 0
    cores = multiprocessing.cpu_count()
    tasks = []

    for idx, v_info in enumerate(videos_info):
        cap = cv2.VideoCapture(v_info['path'])
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        v_info['fps'] = fps
        v_info['total_frames'] = total_frames
        v_info['global_offset'] = total_frames_all

        chunk_size = total_frames // cores
        for i in range(cores):
            start_frame = i * chunk_size
            end_frame = total_frames if i == cores - 1 else (i + 1) * chunk_size

            if start_frame < end_frame:
                tasks.append((
                    idx, v_info['path'], start_frame, end_frame, fps,
                    v_info['coords_map'], DIGITS_SEGMENTS, radius,
                    v_info['preproc_data'], process_every_n,
                    v_info['global_offset']
                ))

        total_frames_all += total_frames

    if total_frames_all == 0:
        return None

    progress_array = multiprocessing.Array('b', total_frames_all)

    pool = multiprocessing.Pool(processes=cores, initializer=init_worker, initargs=(progress_array,))
    async_result = pool.map_async(worker_ocr, tasks)

    win_name = "Concurrent Video Processing"
    cv2.namedWindow(win_name)
    WindowUtils.apply_brief_topmost(win_name)

    num_videos = len(videos_info)
    bar_height = max(15, min(50, 700 // num_videos))
    padding = max(15, min(30, 300 // num_videos))
    ui_height = (bar_height + padding) * num_videos + 40

    while not async_result.ready():
        try:
            progress = np.frombuffer(progress_array.get_obj(), dtype=np.int8)
            ui_frame = np.zeros((ui_height, 800, 3), dtype=np.uint8)

            for idx, v_info in enumerate(videos_info):
                g_offset = v_info['global_offset']
                t_frames = v_info['total_frames']

                if t_frames > 0:
                    v_prog = progress[g_offset: g_offset + t_frames]

                    status_bar = np.zeros((1, t_frames, 3), dtype=np.uint8)
                    status_bar[0, v_prog == 0] = [0, 0, 255]
                    status_bar[0, v_prog == 1] = [0, 255, 0]
                    status_bar[0, v_prog == 2] = [100, 100, 100]

                    display_bar = cv2.resize(status_bar, (760, bar_height), interpolation=cv2.INTER_NEAREST)

                    y_offset = 20 + idx * (bar_height + padding)
                    ui_frame[y_offset + 20: y_offset + 20 + bar_height, 20: 780] = display_bar

                    processed_count = np.sum(v_prog != 0)
                    pct = (processed_count / t_frames) * 100

                    fname = v_info['filename']
                    if len(fname) > 60:
                        fname = fname[:57] + "..."

                    cv2.putText(ui_frame, f"{fname} - {pct:.1f}%", (20, y_offset + 12), cv2.FONT_HERSHEY_SIMPLEX,
                                max(0.4, bar_height / 100), (255, 255, 255), 1)

            cv2.imshow(win_name, ui_frame)

            key = cv2.waitKey(100) & 0xFF
            if key == 27:
                print("Processing aborted by user.")
                pool.terminate()
                pool.join()
                cv2.destroyWindow(win_name)
                return None

        except Exception:
            pass

    cv2.destroyWindow(win_name)

    if not async_result.successful():
        print("Error occurred during multiprocessing.")
        pool.close()
        pool.join()
        return None

    all_chunks = async_result.get()
    pool.close()
    pool.join()

    video_results = {i: [] for i in range(num_videos)}
    for v_idx, chunk in all_chunks:
        video_results[v_idx].extend(chunk)

    for i in range(num_videos):
        video_results[i].sort(key=lambda x: x[0])

    return video_results


def main(radius: int = 2, process_every_n: int = 1, merge_output: bool = False):
    load_segments_definition()
    paths = WindowUtils.open_file_dialog(title="Select one or more raw video files", filetypes=[("MP4 files", "*.mp4")],
                                         multiple=True)

    if not paths:
        return

    videos_info = []
    common_output_folder = None

    print("\nLoading Maps and Preprocessing data...")
    for video_path in paths:
        video_folder = os.path.dirname(video_path)
        video_filename = os.path.basename(video_path)
        base_name = os.path.splitext(video_filename)[0]

        if not common_output_folder:
            common_output_folder = os.path.normpath(os.path.join(video_folder, "output"))
            os.makedirs(common_output_folder, exist_ok=True)

        try:
            coords_map = MappingFile.load_coordinates_map(os.path.join(video_folder, 'maps'), video_filename)
        except Exception as e:
            print(f"Could not load mapping for {video_filename}: {e}")
            continue

        preproc_file = os.path.normpath(os.path.join(video_folder, 'preprocessing', f"{base_name}.json"))
        preproc_data = None
        if os.path.exists(preproc_file):
            with open(preproc_file, 'r') as f:
                preproc_data = json.load(f)

        videos_info.append({
            'path': video_path,
            'filename': video_filename,
            'coords_map': coords_map,
            'preproc_data': preproc_data
        })

    if not videos_info:
        print("No valid videos to process.")
        return

    print("\nStarting Concurrent Multiprocessing...")
    video_results = process_multiple_videos(videos_info, radius, process_every_n)

    if not video_results:
        return

    all_merged_results = []

    for i, v_info in enumerate(videos_info):
        res = video_results[i]
        if res:
            if merge_output:
                for row in res:
                    all_merged_results.append((v_info['filename'],) + tuple(row))
            else:
                save_to_csv(common_output_folder, v_info['filename'], res)

    if merge_output and all_merged_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_merged_csv(common_output_folder, f"mergedfile_{timestamp}", all_merged_results)


if __name__ == '__main__':
    main()