import cv2, tkinter as tk, sys, os
import numpy as np
from tkinter import filedialog, simpledialog, messagebox
from typing import List, Union
from contextlib import contextmanager

# Global tracker to ensure windows only pop to the front ONCE per lifecycle
_TOPMOST_TRACKER = set()


# GET SCREEN SIZE
def get_screen_size():
    try:
        root = tk.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        return w, h
    except:
        return 1920, 1080


SCREEN_WIDTH, SCREEN_HEIGHT = get_screen_size()


# APPLY BRIEF TOPMOST - Forces an OpenCV window to the front briefly, then releases the lock
def apply_brief_topmost(winname, force=False):
    if force or winname not in _TOPMOST_TRACKER:
        try:
            cv2.setWindowProperty(winname, cv2.WND_PROP_TOPMOST, 1)
            cv2.waitKey(1)
            cv2.setWindowProperty(winname, cv2.WND_PROP_TOPMOST, 0)
            _TOPMOST_TRACKER.add(winname)
        except cv2.error:
            pass


# SHOW CENTERED WINDOW
def show_centered_window(winname, img=None, width=800, height=600, fullscreen=False):
    if fullscreen:
        cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(winname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        if img is not None:
            cv2.namedWindow(winname, cv2.WINDOW_AUTOSIZE)
            h, w = img.shape[:2]
        else:
            cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(winname, width, height)
            h, w = height, width

        x = max(0, (SCREEN_WIDTH - w) // 2)
        y = max(0, (SCREEN_HEIGHT - h) // 2)
        cv2.moveWindow(winname, x, y)

    if img is not None:
        cv2.imshow(winname, img)

    apply_brief_topmost(winname)


# ENFORCE WINDOW ALIVE
def enforce_window_alive(winname):
    try:
        if cv2.getWindowProperty(winname, cv2.WND_PROP_VISIBLE) < 1:
            print(f"Window closed. Returning to Hub.")
            sys.exit(0)
    except cv2.error:
        sys.exit(0)


# OPEN FILE DIALOG
def open_file_dialog(title="Select File(s)", filetypes=[("All files", "*.*")], multiple=False) -> Union[str, List[str]]:
    root = tk.Tk()
    root.withdraw()

    root.attributes('-topmost', True)
    root.after(100, lambda: root.attributes('-topmost', False))

    if multiple:
        file_paths = filedialog.askopenfilenames(title=title, filetypes=filetypes)
        result = list(file_paths) if file_paths else []
    else:
        file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        result = file_path if file_path else ""

    root.destroy()
    return result


# PROMPT STRING
def prompt_string(title, prompt):
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    root.after(100, lambda: root.attributes('-topmost', False))

    result = simpledialog.askstring(title=title, prompt=prompt, parent=root)
    root.destroy()
    return result


# PROMPT YES NO
def prompt_yes_no(title, message):
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    root.after(100, lambda: root.attributes('-topmost', False))

    result = messagebox.askyesno(title=title, message=message, parent=root)
    root.destroy()
    return result


# PROMPT INTEGER
def prompt_integer(title, prompt, minvalue=1, maxvalue=100):
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    root.after(100, lambda: root.attributes('-topmost', False))

    result = simpledialog.askinteger(title=title, prompt=prompt, parent=root, minvalue=minvalue, maxvalue=maxvalue)
    root.destroy()
    return result


# VIDEO CAPTURE
@contextmanager
def VideoCapture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


# CENTER TKINTER WINDOW
def center_tkinter_window(root: tk.Tk, min_width: int = 0, min_height: int = 0):
    root.update_idletasks()
    req_w = max(min_width, root.winfo_reqwidth())
    req_h = max(min_height, root.winfo_reqheight())
    x = max(0, (SCREEN_WIDTH - req_w) // 2)
    y = max(0, (SCREEN_HEIGHT - req_h) // 2)
    root.geometry(f"{req_w}x{req_h}+{x}+{y}")


# RESIZE FOR DISPLAY
def resize_for_display(img, width=300):
    h, w = img.shape[:2]
    ratio = width / float(w)
    new_h = int(h * ratio)
    return cv2.resize(img, (width, new_h), interpolation=cv2.INTER_AREA)


# ADD LABEL
def add_label(img, text, font_scale=0.35, thickness=1, bar_height=35):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    labeled_img = cv2.copyMakeBorder(img, bar_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    text_y = int(bar_height * 0.7)
    cv2.putText(labeled_img, text, (5, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    return labeled_img


# CREATE DISPLAY CANVAS
def create_display_canvas(frame, text, font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.9, color=(0, 0, 255), thickness=2):
    screen_w, screen_h = SCREEN_WIDTH, SCREEN_HEIGHT
    available_h = screen_h - 100
    available_w = screen_w

    h, w = frame.shape[:2]
    scale_factor = min(available_w / w, available_h / h)

    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    resized_frame = cv2.resize(frame, (new_w, new_h))

    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)

    offset_x = (screen_w - new_w) // 2
    offset_y = 100 + (available_h - new_h) // 2

    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized_frame
    cv2.putText(canvas, text, (30, 60), font, scale, color, thickness, cv2.LINE_AA)

    return canvas, scale_factor, offset_x, offset_y