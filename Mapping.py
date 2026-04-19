import cv2, math, tkinter as tk, numpy as np, json, os
from tkinter import simpledialog
from typing import List, Tuple
import WindowUtils

FRAME = np.ndarray
COORDINATES = Tuple[int, int]
SEGMENTS_COORDINATES = List[COORDINATES]
COORDINATES_MAP = List[SEGMENTS_COORDINATES]

COLOR = Tuple[int, int, int]
RED = (0, 0, 255)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)

TEXT_COLOR = RED
TEXT_THICKNESS = 2
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.9


# GET SEGMENTS COUNT - Safely loads the segment count (N) from the JSON definition file
# Inputs: None; Outputs: int representing the number of segments per digit
def get_segments_count() -> int:
    map_file = 'segments_map.json'
    if not os.path.exists(map_file):
        map_file = 'segments_map_default.json'
        if not os.path.exists(map_file):
            return 7  # Safe fallback to standard 7-segment display

    with open(map_file, 'r') as f:
        data = json.load(f)
        return data.get('segments_count', 7)


# GET ORDINAL - Converts an integer into its ordinal string representation (1st, 2nd, 3rd, etc.)
# Inputs: n (int); Outputs: str representing the ordinal
def get_ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}" + {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')


# RENDER POINT - Draws a center dot and an outer radius ring on the frame
# Inputs: frame (FRAME), coordinates (COORDINATES), color (COLOR), radius (int); Outputs: None
def render_point(frame: 'FRAME', coordinates: COORDINATES, color: COLOR, radius: int = 2) -> None:
    # Draw a tiny solid dot in the exact center
    cv2.circle(frame, coordinates, radius=1, color=color, thickness=-1)

    # Draw the hollow outer boundary ring representing the sampling area
    if radius > 1:
        cv2.circle(frame, coordinates, radius=radius, color=color, thickness=1)


# RENDER POINTS - Iterates over a list of coordinates and draws them on the frame
# Inputs: frame (FRAME), points (List[COORDINATES]), color (COLOR), radius (int); Outputs: None
def render_points(frame: 'FRAME', points: List[COORDINATES], color: COLOR, radius: int = 2) -> None:
    for coordinates in points:
        render_point(frame, coordinates, color, radius)


# PROMPT DIGITS COUNT - Opens a GUI dialog to ask the user for the total number of digits on the display
# Inputs: frame_number (int), frame (FRAME); Outputs: int representing total number of digits
def prompt_digits_count(frame_number: int, frame: 'FRAME') -> int:
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)  # Forces the dialog to appear above OpenCV windows

    while True:
        digits_count = simpledialog.askinteger(
            title="Calibration",
            prompt="How many digits does the display have?",
            parent=root,
            minvalue=1,
            maxvalue=50
        )

        if digits_count is not None:
            root.destroy()
            return digits_count

        print('Digit count is required to proceed.')


# SELECT POINTS - Handles the interactive window and mouse callbacks for users to click and select coordinates
# Inputs: frame (FRAME), count (int), color (COLOR), title (str), radius (int); Outputs: List[COORDINATES] of the clicked points
def select_points(frame: 'FRAME', count: int, color: COLOR, title: str, radius: int = 2) -> List[COORDINATES]:
    result: List[COORDINATES] = []

    win_name = title

    display_canvas, scale, offset_x, offset_y = WindowUtils.create_display_canvas(frame, title, TEXT_FONT, TEXT_SCALE,
                                                                                  TEXT_COLOR, TEXT_THICKNESS)

    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            orig_x = int((x - offset_x) / scale)  # Reverses the canvas translation math to find original X
            orig_y = int((y - offset_y) / scale)  # Reverses the canvas translation math to find original Y

            h, w = frame.shape[:2]
            if 0 <= orig_x < w and 0 <= orig_y < h:  # Ensures the click landed inside the actual video bounds
                coordinates = (orig_x, orig_y)
                result.append(coordinates)

                render_point(frame, coordinates, color, radius)

                new_canvas, _, _, _ = WindowUtils.create_display_canvas(frame, title, TEXT_FONT, TEXT_SCALE, TEXT_COLOR,
                                                                        TEXT_THICKNESS)
                cv2.imshow(win_name, new_canvas)

    WindowUtils.show_centered_window(win_name, display_canvas, fullscreen=True)
    cv2.setMouseCallback(win_name, click_event)

    while len(result) < count:
        cv2.waitKey(10)
        WindowUtils.enforce_window_alive(win_name)

    cv2.destroyWindow(win_name)
    return result


# FORECAST DIGITS COORDINATES - Calculates vectors between an anchor digit and the first digit to extrapolate remaining coordinates
# Inputs: first_digit (List[COORDINATES]), anchor_digit (List[COORDINATES]), total_digits (int); Outputs: List[List[COORDINATES]] for all digits
def forecast_digits_coordinates(first_digit: List[COORDINATES], anchor_digit: List[COORDINATES], total_digits: int) -> \
List[List[COORDINATES]]:
    k = math.ceil(total_digits / 2)

    if total_digits <= 2:
        k = total_digits

    steps = k - 1

    if steps == 0:
        steps = 1

    vectors = []

    for (x1, y1), (xk, yk) in zip(first_digit, anchor_digit):
        vector_x = (xk - x1) / steps  # Calculates horizontal distance per digit step
        vector_y = (yk - y1) / steps  # Calculates vertical distance per digit step
        vectors.append((vector_x, vector_y))

    result = []

    for i in range(total_digits):
        digit = []
        for (x1, y1), (xd, yd) in zip(first_digit, vectors):
            exact_x = x1 + xd * i
            exact_y = y1 + yd * i

            digit.append((
                int(round(exact_x)),
                int(round(exact_y))
            ))
        result.append(digit)

    return result


# EXTRACT COORDINATES MAP - The main orchestration function that handles the UI flow for forecasting or manually selecting all digit segments
# Inputs: frame_number (int), frame (FRAME), auto_mode (bool), radius (int), progress_str (str); Outputs: COORDINATES_MAP containing all mapped segments
def extract_coordinates_map(frame_number: int, frame: 'FRAME', auto_mode: bool = True,
                            radius: int = 2, progress_str: str = "") -> COORDINATES_MAP:
    digits_count = prompt_digits_count(frame_number, frame)

    segments_count = get_segments_count()

    frame_original = frame.copy()

    if auto_mode:
        # Correctly calculate the anchor digit for 2-digit displays
        anchor_digit = math.ceil(digits_count / 2)
        if digits_count == 2:
            anchor_digit = 2

        while True:
            frame_preview = frame_original.copy()

            instr = f'{progress_str} Pick {segments_count} points representing segments of the RIGHTMOST digit.'.strip()
            first_digit = select_points(frame_preview, segments_count, RED, instr, radius)
            render_points(frame_preview, first_digit, BLUE, radius)

            # Skip the second prompt entirely if there is only 1 digit on the display
            if digits_count == 1:
                other_digits = [first_digit]
            else:
                ordinal_str = get_ordinal(anchor_digit)
                instr = f'{progress_str} Pick {segments_count} points representing segments of the {ordinal_str} digit from the RIGHT.'.strip()
                second_digit = select_points(frame_preview, segments_count, RED, instr, radius)
                render_points(frame_preview, second_digit, BLUE, radius)

                other_digits = forecast_digits_coordinates(first_digit, second_digit, digits_count)

            for digit in other_digits:
                render_points(frame_preview, digit, CYAN, radius)

            title = f"{progress_str} RESULT: Press ENTER or SPACE to Continue, S to Select All Manually, T to Retry Auto".strip()
            final_display_canvas, _, _, _ = WindowUtils.create_display_canvas(frame_preview, title, TEXT_FONT,
                                                                              TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)
            WindowUtils.show_centered_window("RESULT", final_display_canvas, fullscreen=True)

            while True:
                key = cv2.waitKey(0) & 0xFF  # 0xFF masks to 8-bit for cross-platform compatibility
                if key in (13, 32):  # Enter, Space
                    method = 'c'
                    break
                elif key in (ord('t'), ord('T')):
                    method = 't'
                    break
                elif key in (ord('s'), ord('S')):
                    method = 's'
                    break
                WindowUtils.enforce_window_alive("RESULT")
            cv2.destroyWindow("RESULT")

            if method == 'c':
                return other_digits
            elif method == 't':
                continue
            elif method == 's':
                break  # Escape the auto loop and fall through to manual mode

    # --- Manual Selection Mode ---
    instr = f"{progress_str} Manual Selection mode".strip()

    manual_frame = frame_original.copy()
    total_points_needed = digits_count * segments_count

    all_manual_points = select_points(
        manual_frame,
        total_points_needed,
        RED,
        instr,
        radius
    )

    final_map = []
    for i in range(0, len(all_manual_points), segments_count):
        final_map.append(all_manual_points[i: i + segments_count])

    return final_map