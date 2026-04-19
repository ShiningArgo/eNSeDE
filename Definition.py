import cv2, json, math, os
import numpy as np
import WindowUtils


def draw_outlined_text(img, text, pos):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    cv2.putText(img, text, pos, font, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, scale, (255, 255, 255), 1, cv2.LINE_AA)


def save_reference_image(frame, points, filename='input_order.png'):
    ref_frame = frame.copy()
    for i, (px, py) in enumerate(points):
        cv2.circle(ref_frame, (px, py), 4, (0, 0, 255), -1)
        draw_outlined_text(ref_frame, str(i), (px + 10, py + 10))

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    padding = 40
    min_x = max(0, min(xs) - padding)
    max_x = min(ref_frame.shape[1], max(xs) + padding)
    min_y = max(0, min(ys) - padding)
    max_y = min(ref_frame.shape[0], max(ys) + padding)

    cropped_img = ref_frame[min_y:max_y, min_x:max_x]

    cv2.imwrite(filename, cropped_img)
    print(f"\nSaved segment reference image to '{filename}'")


def generate_default_map():
    default_mappings = {
        "0111111": "0",
        "0000110": "1",
        "1011011": "2",
        "1001111": "3",
        "1100110": "4",
        "1101101": "5",
        "1111101": "6",
        "0000111": "7",
        "1111111": "8",
        "1101111": "9",
        "0000000": ""
    }

    default_data = {
        "segments_count": 7,
        "mappings": default_mappings
    }

    with open('segments_map_default.json', 'w') as f:
        json.dump(default_data, f, indent=4)
    print("Generated 'segments_map_default.json'")


def prompt_segment_count() -> int:
    while True:
        count = WindowUtils.prompt_integer(
            title="N-Segment Definition",
            prompt="How many segments does your display have (e.g., 7, 14, 16)?",
            minvalue=1,
            maxvalue=100
        )
        if count is not None:
            return count
        print("Segment count is required to proceed.")


def choose_frame_from_media(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        frame = cv2.imread(file_path)
        if frame is None: print(f"Error: Could not open image at {file_path}")
        return frame

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {file_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    window_name = "Press ENTER to select a clear frame for mapping."

    cv2.namedWindow(window_name)
    WindowUtils.apply_brief_topmost(window_name)

    current_frame = [None]

    def on_trackbar(val):
        cap.set(cv2.CAP_PROP_POS_FRAMES, val)
        ret, frame = cap.read()
        if ret:
            current_frame[0] = frame
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Frame: {val}/{max(1, total_frames - 1)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
            cv2.imshow(window_name, display_frame)

    cv2.createTrackbar("Position", window_name, 0, max(0, total_frames - 1), on_trackbar)
    on_trackbar(0)

    selected_frame = None
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32):
            selected_frame = current_frame[0]
            break
        elif key == 27:
            print("Canceled frame selection.")
            break
        WindowUtils.enforce_window_alive(window_name)

    cv2.destroyWindow(window_name)
    cap.release()

    return selected_frame


def select_reference_points(frame, n: int):
    points = []
    win_name = f"1. Define Segment Order (Click {n} segments in order)"

    canvas = frame.copy()
    cv2.rectangle(canvas, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(canvas, f"Click {n} segments in order. ({len(points)}/{n})",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < n:
            points.append((x, y))

            draw_canvas = frame.copy()
            cv2.rectangle(draw_canvas, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(draw_canvas, f"Click {n} segments in order. ({len(points)}/{n})",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            for i, (px, py) in enumerate(points):
                cv2.circle(draw_canvas, (px, py), 4, (0, 0, 255), -1)
                draw_outlined_text(draw_canvas, str(i), (px + 10, py + 10))

            cv2.imshow(win_name, draw_canvas)

    WindowUtils.show_centered_window(win_name, canvas, fullscreen=False)
    cv2.setMouseCallback(win_name, click_event)

    while len(points) < n:
        cv2.waitKey(10)
        WindowUtils.enforce_window_alive(win_name)

    cv2.destroyWindow(win_name)
    return points


def define_symbols(frame, points, n: int):
    mappings = {}
    win_name = "2. Toggle Segments (Click dots to toggle) & Press ENTER to Map"

    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    h, w = frame.shape[:2]
    panel_w = 300
    extended_w = w + panel_w
    base_h = max(h, 400)  # Ensure window is tall enough to hold a list

    # Create the extended canvas
    base_extended_frame = np.zeros((base_h, extended_w, 3), dtype=np.uint8)
    base_extended_frame[:h, :w] = frame

    WindowUtils.show_centered_window(win_name, base_extended_frame, fullscreen=False)

    while True:
        segment_states = [False] * n

        def draw_state():
            draw_canvas = base_extended_frame.copy()

            cv2.rectangle(draw_canvas, (0, 0), (extended_w, 40), (0, 0, 0), -1)
            cv2.putText(draw_canvas, "Click dots to toggle ON (Green) / OFF (Red). Press ENTER to map.",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            for i, (px, py) in enumerate(points):
                color = (0, 255, 0) if segment_states[i] else (0, 0, 255)
                cv2.circle(draw_canvas, (px, py), 6, color, -1)
                draw_outlined_text(draw_canvas, str(i), (px + 10, py + 10))

            cv2.rectangle(draw_canvas, (w, 40), (extended_w, base_h), (30, 30, 30), -1)
            cv2.line(draw_canvas, (w, 40), (w, base_h), (100, 100, 100), 2)

            cv2.putText(draw_canvas, "Current Mappings:", (w + 15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                        2)
            cv2.line(draw_canvas, (w + 15, 85), (extended_w - 15, 85), (100, 100, 100), 1)

            y_offset = 110
            for bin_key, char in mappings.items():

                # Enforce max character length just in case someone types a huge word
                display_char = char[:6] + ".." if len(char) > 8 else char

                # DRAWING AS A FIXED GRID (Symbol, Equals Sign, Binary String)
                cv2.putText(draw_canvas, f"'{display_char}'", (w + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (200, 255, 200), 1, cv2.LINE_AA)
                cv2.putText(draw_canvas, "=", (w + 95, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1,
                            cv2.LINE_AA)
                cv2.putText(draw_canvas, bin_key, (w + 120, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200),
                            1, cv2.LINE_AA)

                y_offset += 30
                if y_offset > base_h - 20:
                    cv2.putText(draw_canvas, "...", (w + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200),
                                1, cv2.LINE_AA)
                    break

            cv2.imshow(win_name, draw_canvas)

        def click_event(event, x, y, flags, params):
            if event == cv2.EVENT_LBUTTONDOWN:
                if x > w:
                    return

                closest_idx = -1
                min_dist = float('inf')

                for i, (px, py) in enumerate(points):
                    dist = math.hypot(x - px, y - py)
                    if dist < min_dist and dist < 20:
                        min_dist = dist
                        closest_idx = i

                if closest_idx != -1:
                    segment_states[closest_idx] = not segment_states[closest_idx]
                    draw_state()

        cv2.setMouseCallback(win_name, click_event)
        draw_state()

        while True:
            key = cv2.waitKey(10) & 0xFF
            if key in (13, 32):
                break
            WindowUtils.enforce_window_alive(win_name)

        binary_result = 0
        for i, state in enumerate(segment_states):
            if state:
                binary_result |= (1 << i)

        cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 0)

        mapped_char = WindowUtils.prompt_string(
            title="Map Combination",
            prompt="What character/string does this segment combination represent?"
        )

        WindowUtils.apply_brief_topmost(win_name, force=True)

        if mapped_char is not None:
            bin_key = format(binary_result, f'0{n}b')

            if bin_key in mappings:
                print(f"Overwrote existing mapping for {bin_key}: '{mappings[bin_key]}' is now '{mapped_char}'")
            else:
                print(f"Mapped binary {bin_key} to '{mapped_char}'")

            mappings[bin_key] = mapped_char
            draw_state()

        continue_mapping = WindowUtils.prompt_yes_no(
            title="Continue?",
            message="Do you want to define another symbol?"
        )

        WindowUtils.apply_brief_topmost(win_name, force=True)

        if not continue_mapping:
            cv2.destroyWindow(win_name)
            break

    return mappings


def main():
    if not os.path.exists('segments_map_default.json'):
        generate_default_map()

    video_file = WindowUtils.open_file_dialog(title="Select a Video File", filetypes=[("MP4 files", "*.mp4"),
                                                                                      ("Image files",
                                                                                       "*.jpg *.png *.jpeg *.bmp")])
    if not video_file:
        print("No video selected. Exiting.")
        return

    frame = choose_frame_from_media(video_file)
    if frame is None:
        print("No frame selected. Exiting.")
        return

    n = prompt_segment_count()

    points = select_reference_points(frame, n)
    if len(points) < n:
        print("Did not finish selecting points. Exiting.")
        return

    save_reference_image(frame, points)

    mappings = define_symbols(frame, points, n)

    if mappings:
        data = {
            "segments_count": n,
            "mappings": mappings
        }
        with open('segments_map.json', 'w') as f:
            json.dump(data, f, indent=4)
        print("\nSuccessfully saved custom definitions to 'segments_map.json'!")
    else:
        print("\nNo mappings were created.")


if __name__ == '__main__':
    main()