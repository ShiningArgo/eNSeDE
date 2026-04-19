import cv2, json, numpy as np, copy, os
import WindowUtils

_CACHED_CONFIG = None


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


def get_config():
    global _CACHED_CONFIG
    if _CACHED_CONFIG is None:
        default_params = {
            "global_rotation": 0, "dark_on_light": True, "auto_contrast": False, "use_adaptive": True, "use_otsu": True,
            "manual_thresh": 127, "brightness": 100, "contrast": 100, "gamma": 100, "return_bgr": False,
            "blur_ksize": 25, "adapt_blocksize": 179, "adapt_C": 23, "dilate_kernel": (3, 7),
            "dilate_iterations": 4, "close_kernel": (7, 7), "erode_kernel": (3, 3), "erode_iterations": 1
        }
        try:
            with open('config.json', 'r') as file:
                params = json.load(file)
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            try:
                with open('config_default.json', 'r') as file:
                    params = json.load(file)
                print("Notice: 'config.json' not found. Loaded 'config_default.json'.")
            except (FileNotFoundError, json.decoder.JSONDecodeError):
                print("Notice: 'config_default.json' is missing. Creating a new one automatically.")
                params = default_params.copy()
                with open('config_default.json', 'w') as file:
                    json.dump(params, file, indent=4)

        for k, v in default_params.items():
            if k not in params: params[k] = v

        params['dilate_kernel'] = tuple(params['dilate_kernel'])
        params['close_kernel'] = tuple(params['close_kernel'])
        params['erode_kernel'] = tuple(params['erode_kernel'])

        params['k_dilate'] = cv2.getStructuringElement(cv2.MORPH_RECT, params['dilate_kernel'])
        params['k_close'] = cv2.getStructuringElement(cv2.MORPH_RECT, params['close_kernel'])
        params['k_erode'] = cv2.getStructuringElement(cv2.MORPH_RECT, params['erode_kernel'])

        _CACHED_CONFIG = params

    return _CACHED_CONFIG.copy()


def modify_frame(frame, dark_on_light=True, auto_contrast=False, use_adaptive=True, use_otsu=True,
                 manual_thresh=127, brightness=100, contrast=100, gamma=100, return_bgr=False,
                 preview_grayscale=False, blur_ksize=25, adapt_blocksize=179, adapt_C=23,
                 dilate_kernel=(3, 7), dilate_iterations=4, close_kernel=(7, 7), erode_kernel=(3, 3),
                 erode_iterations=1, k_dilate=None, k_close=None, k_erode=None, global_rotation=0):
    if not dark_on_light: frame = cv2.bitwise_not(frame)

    if brightness != 100 or contrast != 100:
        alpha = contrast / 100.0
        beta = brightness - 100
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    if gamma != 100:
        gamma_val = gamma / 100.0
        table = np.array([((i / 255.0) ** gamma_val) * 255 for i in np.arange(0, 256)]).astype("uint8")
        frame = cv2.LUT(frame, table)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if auto_contrast: gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    if preview_grayscale:
        if return_bgr: return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return gray

    if use_adaptive:
        if blur_ksize > 1: gray = cv2.medianBlur(gray, blur_ksize)
        thresh_inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                           adapt_blocksize, adapt_C)
    else:
        if use_otsu:
            _, thresh_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            thresh_val = max(0, min(255, manual_thresh))
            _, thresh_inv = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)

    if k_dilate is None: k_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, dilate_kernel)
    dilated = cv2.dilate(thresh_inv, k_dilate, iterations=dilate_iterations)

    if k_close is None: k_close = cv2.getStructuringElement(cv2.MORPH_RECT, close_kernel)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, k_close)

    if erode_iterations > 0:
        if k_erode is None: k_erode = cv2.getStructuringElement(cv2.MORPH_RECT, erode_kernel)
        cleaned_foreground = cv2.erode(closed, k_erode, iterations=erode_iterations)
    else:
        cleaned_foreground = closed

    final = cv2.bitwise_not(cleaned_foreground)
    if return_bgr: final = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)
    return final


def clean_frame(frame, **kwargs):
    params = get_config()
    params.update(kwargs)
    return modify_frame(frame, **params)


def tune_visually(cap, total_frames, preproc_data):
    params = get_config()
    baseline_params = copy.deepcopy(params)

    app_state = {'frame_idx': 0, 'frame': None, 'needs_redraw': True}

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, initial = cap.read()
    app_state['frame'] = apply_preprocessing(initial, preproc_data)

    def on_trackbar(val):
        cap.set(cv2.CAP_PROP_POS_FRAMES, val)
        r, f = cap.read()
        if r:
            app_state['frame'] = apply_preprocessing(f, preproc_data)
            app_state['frame_idx'] = val
            app_state['needs_redraw'] = True

    def tune_binary_step(param_name, title, desc_false, desc_true, kw_override_false=None, kw_override_true=None,
                         preview_grayscale=False):
        win_name = f"{title} (Backspace to go back)"
        cv2.namedWindow(win_name)
        WindowUtils.apply_brief_topmost(win_name)

        cv2.createTrackbar("Frame", win_name, app_state['frame_idx'], total_frames - 1, on_trackbar)
        action = [None]

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                action[0] = (x, y)

        cv2.setMouseCallback(win_name, click_event)
        app_state['needs_redraw'] = True

        img_w = 600
        top_h = 0

        while True:
            if app_state['needs_redraw']:
                current_frame = app_state['frame']
                kw_false, kw_true, kw_curr = params.copy(), params.copy(), params.copy()
                kw_false["preview_grayscale"] = kw_true["preview_grayscale"] = kw_curr[
                    "preview_grayscale"] = preview_grayscale

                if kw_override_false:
                    kw_false.update(kw_override_false)
                else:
                    kw_false[param_name] = False

                if kw_override_true:
                    kw_true.update(kw_override_true)
                else:
                    kw_true[param_name] = True

                img_false = modify_frame(current_frame, **kw_false)
                img_true = modify_frame(current_frame, **kw_true)
                img_curr = modify_frame(current_frame, **kw_curr)

                d_false = WindowUtils.add_label(WindowUtils.resize_for_display(img_false, img_w),
                                                f"Click to set: {desc_false}", 0.5, 1)
                d_true = WindowUtils.add_label(WindowUtils.resize_for_display(img_true, img_w),
                                               f"Click to set: {desc_true}", 0.5, 1)

                top_row = np.hstack((d_false, d_true))
                top_h, top_w = top_row.shape[:2]

                large_current = WindowUtils.resize_for_display(img_curr, top_w)

                is_true_active = all(params.get(k) == v for k, v in kw_override_true.items()) if kw_override_true else \
                    params[param_name]
                curr_desc = desc_true if is_true_active else desc_false
                large_current = WindowUtils.add_label(large_current, f"CURRENT: {curr_desc} (Click here to CONFIRM)",
                                                      0.6, 2)

                full_ui = np.vstack((top_row, large_current))
                WindowUtils.show_centered_window(win_name, full_ui, fullscreen=False)
                app_state['needs_redraw'] = False

            if action[0] is not None:
                x, y = action[0]
                action[0] = None

                if y < top_h:
                    if x < img_w:
                        if kw_override_false:
                            params.update(kw_override_false)
                        else:
                            params[param_name] = False
                    else:
                        if kw_override_true:
                            params.update(kw_override_true)
                        else:
                            params[param_name] = True
                    app_state['needs_redraw'] = True
                else:
                    cv2.destroyWindow(win_name)
                    return 1

            key = cv2.waitKey(20) & 0xFF
            if key == 8:
                cv2.destroyWindow(win_name)
                return -1
            WindowUtils.enforce_window_alive(win_name)

    def get_post_lighting_batches():
        steps = []
        if params["use_adaptive"]:
            steps.extend([("blur_ksize", 2, True, None, 1), ("adapt_blocksize", 2, True, None, 3),
                          ("adapt_C", 2, False, None, 0)])
        else:
            if not params["use_otsu"]: steps.extend([("manual_thresh", 5, False, None, 0)])
        steps.extend([("dilate_iterations", 1, False, None, 0), ("dilate_kernel", 2, True, 0, 1),
                      ("dilate_kernel", 2, True, 1, 1), ("close_kernel", 2, True, 0, 1),
                      ("close_kernel", 2, True, 1, 1), ("erode_iterations", 1, False, None, 0),
                      ("erode_kernel", 2, True, 0, 1), ("erode_kernel", 2, True, 1, 1)])
        return [steps]

    def tune_batch(batch, step_name, preview_grayscale=False):
        win_name = f"{step_name} (Backspace to go back)"
        cv2.namedWindow(win_name)
        WindowUtils.apply_brief_topmost(win_name)

        cv2.createTrackbar("Frame", win_name, app_state['frame_idx'], total_frames - 1, on_trackbar)
        action = [None]

        def click_event(event, x, y, flags, params_dict):
            if event == cv2.EVENT_LBUTTONDOWN:
                action[0] = (x, y)

        cv2.setMouseCallback(win_name, click_event)
        app_state['needs_redraw'] = True

        avail_h = WindowUtils.SCREEN_HEIGHT - 100
        orig_h, orig_w = app_state['frame'].shape[:2]
        aspect = orig_h / float(orig_w)

        target_row_height = (avail_h * 0.7) / len(batch)
        img_w = max(80, min(250, int((target_row_height - 25) / aspect)))
        font_scale = max(0.25, img_w / 600.0)

        param_grid_h = 0
        variants = []

        while True:
            if app_state['needs_redraw']:
                current_frame = app_state['frame']
                rows_imgs, variants = [], []
                norm_kw = params.copy()
                norm_kw["preview_grayscale"] = preview_grayscale
                shared_norm_img = modify_frame(current_frame, **norm_kw)

                for i, (p_name, step, is_odd, t_idx, min_v) in enumerate(batch):
                    current_val = params[p_name]
                    baseline_val = baseline_params[p_name][t_idx] if t_idx is not None else baseline_params[p_name]
                    v = current_val[t_idx] if t_idx is not None else current_val

                    delta_pct = max(int(abs(v) * 0.5), step * 2)
                    if is_odd and delta_pct % 2 != 0: delta_pct += 1

                    def constrain(val):
                        val = max(min_v, val)
                        if p_name == "manual_thresh": val = min(255, val)
                        if is_odd and val % 2 == 0: val += 1
                        return val

                    v_lp, v_mlp, v_ls, v_ms, v_mmp, v_mp = constrain(v - delta_pct), constrain(
                        v - (delta_pct + step) // 2), constrain(v - step), constrain(v + step), constrain(
                        v + (delta_pct + step) // 2), constrain(v + delta_pct)
                    variants.append((v_lp, v_mlp, v_ls, v_ms, v_mmp, v_mp))

                    def build_kwargs(val):
                        kw = params.copy()
                        kw["preview_grayscale"] = preview_grayscale
                        if t_idx is not None:
                            curr_tup = list(kw[p_name])
                            curr_tup[t_idx] = val
                            kw[p_name] = tuple(curr_tup)
                        else:
                            kw[p_name] = val
                        return kw

                    imgs = [modify_frame(current_frame, **build_kwargs(val)) for val in
                            [v_lp, v_mlp, v_ls, v_ms, v_mmp, v_mp]]
                    labels = ["-%", "-Mid", "-Stp", "+Stp", "+Mid", "+%"]

                    display_name = p_name
                    if t_idx == 0: display_name += " (X)"
                    if t_idx == 1: display_name += " (Y)"

                    row_labeled = []
                    for idx, img in enumerate(imgs):
                        val = variants[-1][idx]
                        txt = f"{labels[idx]} ({val})" + (" [Def]" if val == baseline_val else "")
                        row_labeled.append(
                            WindowUtils.add_label(WindowUtils.resize_for_display(img, img_w), f"{display_name}: {txt}",
                                                  font_scale=font_scale, thickness=1, bar_height=25))

                    rows_imgs.append(np.hstack(row_labeled))

                param_grid = np.vstack(rows_imgs)
                param_grid_h, param_grid_w = param_grid.shape[:2]

                target_h_rem = avail_h - param_grid_h
                if target_h_rem > 50:
                    large_scale = min(param_grid_w / float(orig_w), target_h_rem / float(orig_h))
                    large_w, large_h = int(orig_w * large_scale), int(orig_h * large_scale)

                    large_current_raw = WindowUtils.add_label(
                        cv2.resize(shared_norm_img, (large_w, large_h), interpolation=cv2.INTER_AREA),
                        "CURRENT SETTINGS (Click here to CONFIRM)", font_scale=0.6, thickness=2)

                    pad_l = max(0, (param_grid_w - large_current_raw.shape[1]) // 2)
                    pad_r = max(0, param_grid_w - large_current_raw.shape[1] - pad_l)
                    large_current = cv2.copyMakeBorder(large_current_raw, 0, 0, pad_l, pad_r, cv2.BORDER_CONSTANT,
                                                       value=(0, 0, 0))

                    full_ui = np.vstack((param_grid, large_current))
                else:
                    full_ui = param_grid

                WindowUtils.show_centered_window(win_name, full_ui, fullscreen=False)
                app_state['needs_redraw'] = False

            if action[0] is not None:
                x, y = action[0]
                action[0] = None

                if y < param_grid_h:
                    row_idx, col_idx = y // (param_grid_h // len(batch)), x // img_w
                    if row_idx < len(batch) and col_idx < 6:
                        p_name, step, is_odd, t_idx, min_v = batch[row_idx]
                        val = variants[row_idx][col_idx]
                        if t_idx is not None:
                            curr_tup = list(params[p_name])
                            curr_tup[t_idx] = val
                            params[p_name] = tuple(curr_tup)
                        else:
                            params[p_name] = val
                        app_state['needs_redraw'] = True
                else:
                    cv2.destroyWindow(win_name)
                    return 1

            key = cv2.waitKey(20) & 0xFF
            if key == 8:
                cv2.destroyWindow(win_name)
                return -1
            WindowUtils.enforce_window_alive(win_name)

    def tune_dark_on_light():
        return tune_binary_step("dark_on_light", "Tuning: Background Type", "Light Numbers on Dark Bg",
                                "Dark Numbers on Light Bg")

    def tune_auto_contrast():
        return tune_binary_step("auto_contrast", "Tuning: Auto-Contrast", "Standard", "Auto-Stretch (Min-Max)",
                                preview_grayscale=True)

    def tune_threshold_method():
        return tune_binary_step("use_adaptive", "Tuning: Thresholding Method", "Simple (Global)", "Adaptive (Local)")

    def tune_otsu_or_manual():
        return tune_binary_step(None, "Tuning: Simple Threshold Type", "Manual Threshold", "Otsu (Automatic)",
                                kw_override_false={"use_adaptive": False, "use_otsu": False},
                                kw_override_true={"use_adaptive": False, "use_otsu": True})

    state = 0
    while True:
        if state == 0:
            state += tune_dark_on_light()
        elif state == 1:
            state += tune_auto_contrast()
        elif state == 2:
            state += tune_batch(
                [("brightness", 10, False, None, 0), ("contrast", 10, False, None, 0), ("gamma", 10, False, None, 10)],
                "Tuning: Lighting", preview_grayscale=True)
        elif state == 3:
            state += tune_threshold_method()
        elif state == 4:
            state += tune_otsu_or_manual() if not params["use_adaptive"] else 1
        elif state >= 5:
            batches = get_post_lighting_batches()
            batch_idx = state - 5
            if batch_idx >= len(batches): break
            state += tune_batch(batches[batch_idx], "Tuning: Threshold & Morphology", preview_grayscale=False)
        if state < 0:
            cv2.destroyAllWindows()
            return

    save_params = params.copy()
    save_params['dilate_kernel'] = list(save_params['dilate_kernel'])
    save_params['close_kernel'] = list(save_params['close_kernel'])
    save_params['erode_kernel'] = list(save_params['erode_kernel'])

    save_params['global_rotation'] = preproc_data['rotation_angle']

    save_params.pop('k_dilate', None)
    save_params.pop('k_close', None)
    save_params.pop('k_erode', None)
    save_params.pop('preview_grayscale', None)

    with open('config.json', 'w') as file:
        json.dump(save_params, file, indent=4)

    print("\nTuning complete! Master parameters saved to config.json.")


def setup_master_tuning(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, first_frame = cap.read()
    if not ret: return

    (orig_h, orig_w) = first_frame.shape[:2]
    center = (orig_w // 2, orig_h // 2)

    config = get_config()
    rotation_angle = config.get("global_rotation", 0)

    state = {'frame': first_frame.copy()}

    win_rot = "1. Master Setup: Scrub & Rotate (Press 'R'/'L', ENTER to accept)"
    cv2.namedWindow(win_rot)
    WindowUtils.apply_brief_topmost(win_rot)

    def on_trackbar(val):
        cap.set(cv2.CAP_PROP_POS_FRAMES, val)
        r, f = cap.read()
        if r: state['frame'] = f

    cv2.createTrackbar("Frame", win_rot, 0, total_frames - 1, on_trackbar)

    while True:
        frame_to_process = state['frame']

        if rotation_angle % 360 != 0:
            M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
            current_frame = cv2.warpAffine(frame_to_process, M, (orig_w, orig_h))
        else:
            current_frame = frame_to_process.copy()

        display_frame = current_frame.copy()
        cv2.putText(display_frame, "Scrub to find a good frame. Press 'R' / 'L' to Rotate", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press ENTER or SPACE to accept and move to Cropping", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)
        cv2.putText(display_frame, f"Global Angle: {-rotation_angle} degrees", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2)

        cv2.imshow(win_rot, display_frame)

        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32):
            break
        elif key in (ord('r'), ord('R')):
            rotation_angle -= 5
        elif key in (ord('l'), ord('L')):
            rotation_angle += 5
        WindowUtils.enforce_window_alive(win_rot)

    cv2.destroyWindow(win_rot)

    win_crop = "2. Master Setup: Temporary Crop (Drag box, press ENTER)"
    WindowUtils.show_centered_window(win_crop, current_frame)
    roi = cv2.selectROI(win_crop, current_frame, showCrosshair=True, fromCenter=False, printNotice=False)
    cv2.destroyWindow(win_crop)

    x, y, w, h = roi
    if w == 0 or h == 0:
        x, y, w, h = 0, 0, orig_w, orig_h

    preproc_data = {
        "rotation_angle": rotation_angle,
        "center": center,
        "roi": [x, y, w, h]
    }

    tune_visually(cap, total_frames, preproc_data)
    cap.release()


def main():
    media_file = WindowUtils.open_file_dialog(title="Select a Sample Video for Master Tuning",
                                              filetypes=[("Media files", "*.mp4 *.jpg *.jpeg *.png")], multiple=False)

    if not media_file:
        print("No file selected. Exiting.")
        return

    setup_master_tuning(media_file)


if __name__ == "__main__":
    main()