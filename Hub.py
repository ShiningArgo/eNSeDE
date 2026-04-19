import tkinter as tk, os, cv2, base64, multiprocessing, json
from tkinter import font
import WindowUtils, Definition, Tuning, Calibration, NumRecognition


# HUB CONFIGURATION MANAGEMENT
def load_hub_config():
    default_config = {
        "auto_mode": True,
        "merge_output": False,
        "radius": 2,
        "skip_frames": 1
    }
    try:
        with open('hub_config.json', 'r') as f:
            loaded_config = json.load(f)
            default_config.update(loaded_config)
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        pass
    return default_config


def save_hub_config(root, *args):
    try:
        config = {
            "auto_mode": root.auto_mode_var.get(),
            "merge_output": root.merge_output_var.get(),
            "radius": root.radius_var.get(),
            "skip_frames": root.skip_frames_var.get()
        }
        with open('hub_config.json', 'w') as f:
            json.dump(config, f, indent=4)
    except Exception:
        pass


# LAUNCH SCRIPT
def launch_script(root, script_name: str, refresh_ui, auto_mode: bool = True, radius: int = 2, skip_frames: int = 1,
                  merge_output: bool = False):
    root.withdraw()
    root.update()

    print(f"Launched {script_name}...")

    if script_name == "Definition.py":
        p = multiprocessing.Process(target=Definition.main)
    elif script_name == "Tuning.py":
        p = multiprocessing.Process(target=Tuning.main)
    elif script_name == "Calibration.py":
        p = multiprocessing.Process(target=Calibration.main, args=(auto_mode, radius))
    elif script_name == "NumRecognition.py":
        p = multiprocessing.Process(target=NumRecognition.main, args=(radius, skip_frames, merge_output))

    p.start()
    p.join()

    print(f"Finished {script_name}. Restoring Hub.")
    refresh_ui()
    root.deiconify()


# BUILD UI
def build_ui(root):
    for widget in root.winfo_children():
        widget.destroy()

    config = load_hub_config()

    root.auto_mode_var = tk.BooleanVar(value=config["auto_mode"])
    root.merge_output_var = tk.BooleanVar(value=config["merge_output"])
    root.radius_var = tk.IntVar(value=config["radius"])
    root.skip_frames_var = tk.IntVar(value=config["skip_frames"])

    save_callback = lambda *args: save_hub_config(root, *args)
    root.auto_mode_var.trace_add("write", save_callback)
    root.merge_output_var.trace_add("write", save_callback)
    root.radius_var.trace_add("write", save_callback)
    root.skip_frames_var.trace_add("write", save_callback)

    title_font = font.Font(family="Helvetica", size=18, weight="bold")
    step_font = font.Font(family="Helvetica", size=12, weight="bold")
    desc_font = font.Font(family="Helvetica", size=10, slant="italic")

    main_frame = tk.Frame(root)
    main_frame.pack(fill="both", expand=True)

    left_frame = tk.Frame(main_frame)
    left_frame.pack(side="left", fill="y", expand=False, padx=(30, 10), pady=20)

    right_frame = tk.Frame(main_frame)
    right_frame.pack(side="right", fill="both", expand=True, padx=(10, 30), pady=20)

    title_label = tk.Label(left_frame, text="eN-Segmented Display Enumerator", font=title_font)
    title_label.pack(pady=(0, 5))

    subtitle_label = tk.Label(left_frame, text="Select a module to launch:", fg="gray")
    subtitle_label.pack(pady=(0, 20))

    steps = [
        ("Step 0 (Optional): Define Symbols", "Definition.py",
         "Map visual segment combinations to characters."),

        ("Step 1: Setup & Visual Tuning", "Tuning.py",
         "Set global rotation and visually tune parameters on a sample video."),

        ("Step 2: Crop & Map Coordinates", "Calibration.py",
         "Select videos to individually crop and map coordinate points."),

        ("Step 3: Run OCR", "NumRecognition.py",
         "Process the videos using all configured settings and export to CSV.")
    ]

    for step_title, script_name, description in steps:
        btn_frame = tk.Frame(left_frame)
        btn_frame.pack(fill="x", pady=10)

        if script_name == "Calibration.py":
            cmd = lambda s=script_name: launch_script(root, s, lambda: build_ui(root),
                                                      auto_mode=root.auto_mode_var.get(), radius=root.radius_var.get())
        elif script_name == "NumRecognition.py":
            cmd = lambda s=script_name: launch_script(root, s, lambda: build_ui(root), radius=root.radius_var.get(),
                                                      skip_frames=root.skip_frames_var.get(),
                                                      merge_output=root.merge_output_var.get())
        else:
            cmd = lambda s=script_name: launch_script(root, s, lambda: build_ui(root))

        btn = tk.Button(
            btn_frame,
            text=step_title,
            font=step_font,
            bg="#f0f0f0",
            width=30,
            pady=8,
            cursor="hand2",
            command=cmd
        )
        btn.pack(side="top")

        lbl = tk.Label(btn_frame, text=description, font=desc_font, fg="#555555")
        lbl.pack(side="top", pady=(2, 0))

    img_container = tk.Frame(right_frame)
    img_container.pack(side="top", expand=True)

    image_path = 'input_order.png'
    if os.path.exists(image_path):
        try:
            img_cv = cv2.imread(image_path)
            if img_cv is not None:
                h, w = img_cv.shape[:2]
                scale = min(350 / h, 400 / w)
                new_w, new_h = int(w * scale), int(h * scale)
                resized_cv = cv2.resize(img_cv, (new_w, new_h), interpolation=cv2.INTER_AREA)

                _, buffer = cv2.imencode('.png', resized_cv)
                img_b64 = base64.b64encode(buffer).decode('utf-8')
                img = tk.PhotoImage(data=img_b64)
                right_frame.image = img

                img_label = tk.Label(img_container, image=img)
                img_label.pack(side="top", pady=(0, 10))
                text_label = tk.Label(img_container, text="Current input order", font=desc_font, fg="gray")
                text_label.pack(side="top")
        except Exception as e:
            print(f"Notice: Could not load '{image_path}': {e}")

    settings_frame = tk.LabelFrame(right_frame, text="Global Settings", font=step_font, padx=15, pady=15, bg="#ffffff")
    settings_frame.pack(side="bottom", fill="x", pady=(20, 0))

    chk_auto = tk.Checkbutton(settings_frame, text="Automatic mapping mode", variable=root.auto_mode_var, bg="#ffffff")
    chk_auto.pack(anchor="w", pady=(0, 2))

    chk_merge = tk.Checkbutton(settings_frame, text="Merge output of multiple videos into one file",
                               variable=root.merge_output_var, bg="#ffffff")
    chk_merge.pack(anchor="w", pady=(0, 5))

    rad_frame = tk.Frame(settings_frame, bg="#ffffff")
    rad_frame.pack(fill="x", pady=5)
    tk.Label(rad_frame, text="Mapping point radius:", bg="#ffffff").pack(side="left")
    tk.Spinbox(rad_frame, from_=1, to=20, textvariable=root.radius_var, width=5).pack(side="right")

    skip_frame = tk.Frame(settings_frame, bg="#ffffff")
    skip_frame.pack(fill="x", pady=5)
    tk.Label(skip_frame, text="Process every Nth frame:", bg="#ffffff").pack(side="left")
    tk.Spinbox(skip_frame, from_=1, to=1000, textvariable=root.skip_frames_var, width=5).pack(side="right")

    WindowUtils.center_tkinter_window(root, min_width=850, min_height=550)


def main():
    root = tk.Tk()
    root.title("eNSeDE: eN-Segment Display Enumerator")

    # Ensure the main Hub window pops to the front on launch
    root.attributes('-topmost', True)
    root.after(100, lambda: root.attributes('-topmost', False))

    build_ui(root)
    root.mainloop()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()