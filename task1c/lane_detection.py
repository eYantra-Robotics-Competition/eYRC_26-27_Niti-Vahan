'''
*****************************************************************************************
*
*  ===============================================
*     Niti Vahan (NV) Theme of eYRC 2026-27
*  ===============================================
*
*  This script is intended for implementation of Task 1C of Niti Vahan (NV) Theme.
*
*  Filename:         lane_detection.py
*  Created:          2026
*  Last Modified:
*  Author:           e-Yantra Team
*
*  You are ONLY allowed to write your code inside the block marked
*  "ADD YOUR IMPLEMENTATION HERE". Do not change anything outside it - the
*  evaluation script relies on the rest of this file staying as it is.
*
*****************************************************************************************
'''

# Team ID:          < Team-ID >
# Author List:      < Names of the team members who worked on this file, comma separated >
# Filename:         lane_detection.py
# Functions:        detect_lane
# Global variables: < List any global variables you add, "None" if you add none >


####################### IMPORT MODULES #######################
import argparse
import json
import os

import cv2
import numpy as np
##############################################################

# The only three values "lane" is allowed to take.
LANE_LEFT = "left"
LANE_RIGHT = "right"
LANE_UNKNOWN = "unknown"
VALID_LANES = (LANE_LEFT, LANE_RIGHT, LANE_UNKNOWN)

############### ADD YOUR IMPLEMENTATION HERE #################
 
_prev_lane = "unknown"
_prev_center_x = -1
_prev_lane_width = 315.0
 
 
def detect_lane(frame):
    global _prev_lane, _prev_center_x, _prev_lane_width
 
    h, w = frame.shape[:2]
 
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
 
    # Road region
    roi_top = int(0.35 * h)
    roi = hsv[roi_top:h, :]
 
    # White dashed centre line
    white_mask = cv2.inRange(
        roi,
        np.array([0, 0, 140], dtype=np.uint8),
        np.array([180, 115, 255], dtype=np.uint8)
    )
 
    # Yellow outer boundary
    yellow_mask = cv2.inRange(
        roi,
        np.array([12, 55, 65], dtype=np.uint8),
        np.array([45, 255, 255], dtype=np.uint8)
    )
 
    # Morphological cleanup
    close_kernel = np.ones((5, 5), np.uint8)
    open_kernel = np.ones((3, 3), np.uint8)
 
    white_mask = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=2
    )
 
    yellow_mask = cv2.morphologyEx(
        yellow_mask,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=2
    )
 
    white_mask = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_OPEN,
        open_kernel,
        iterations=1
    )
 
    yellow_mask = cv2.morphologyEx(
        yellow_mask,
        cv2.MORPH_OPEN,
        open_kernel,
        iterations=1
    )
 
    # Get pixel coordinates
    wy, wx = np.where(white_mask > 0)
    yy, yx = np.where(yellow_mask > 0)
 
    wy = wy + roi_top
    yy = yy + roi_top
 
    # Remove extreme image-edge noise
    margin = int(0.02 * w)
 
    keep_w = (wx >= margin) & (wx < w - margin)
    keep_y = (yx >= margin) & (yx < w - margin)
 
    wx, wy = wx[keep_w], wy[keep_w]
    yx, yy = yx[keep_y], yy[keep_y]
 
    # Fit curves
    white_fit = _fit_lane_curve(wx, wy)
    yellow_fit = _fit_lane_curve(yx, yy)
 
    # Several look-ahead positions
    y_samples = np.array([
        0.68 * h,
        0.74 * h,
        0.80 * h,
        0.86 * h,
        0.91 * h
    ])
 
    white_vals = None
    yellow_vals = None
 
    if white_fit is not None:
        white_vals = np.polyval(
            white_fit,
            y_samples
        )
 
    if yellow_fit is not None:
        yellow_vals = np.polyval(
            yellow_fit,
            y_samples
        )
 
    # ---------------------------------------------------------
    # Estimate lane width from places where both markings exist
    # ---------------------------------------------------------
    widths = []
 
    if white_vals is not None and yellow_vals is not None:
 
        for white_x, yellow_x in zip(
            white_vals,
            yellow_vals
        ):
            width_here = yellow_x - white_x
 
            if (
                0 <= white_x < w and
                0 <= yellow_x < w and
                0.10 * w < width_here < 0.65 * w
            ):
                widths.append(width_here)
 
    if len(widths) >= 2:
 
        measured_width = float(
            np.median(widths)
        )
 
        _prev_lane_width = (
            0.80 * _prev_lane_width +
            0.20 * measured_width
        )
 
    lane_width = _prev_lane_width
 
    # ---------------------------------------------------------
    # Evaluate at look-ahead position
    # ---------------------------------------------------------
    y_eval = 0.83 * h
 
    white_x = None
    yellow_x = None
 
    if white_fit is not None:
        white_x = float(
            np.polyval(
                white_fit,
                y_eval
            )
        )
 
    if yellow_fit is not None:
        yellow_x = float(
            np.polyval(
                yellow_fit,
                y_eval
            )
        )
 
    white_valid = (
        white_x is not None and
        np.isfinite(white_x) and
        0 <= white_x < w
    )
 
    yellow_valid = (
        yellow_x is not None and
        np.isfinite(yellow_x) and
        0 <= yellow_x < w
    )
 
    # ---------------------------------------------------------
    # Determine lane
    # ---------------------------------------------------------
    vehicle_x = w / 2.0
 
    lane = "unknown"
    center_x = -1
 
    if white_valid:
 
        # White line is to the right of vehicle
        # => vehicle is in LEFT lane
        if white_x > vehicle_x:
 
            lane = "left"
 
            center_x = (
                white_x -
                lane_width / 2.0
            )
 
        # White line is to the left of vehicle
        # => vehicle is in RIGHT lane
        elif white_x < vehicle_x:
 
            lane = "right"
 
            # Best case: both markings visible
            if yellow_valid and yellow_x > white_x:
 
                actual_width = yellow_x - white_x
 
                if (
                    0.10 * w <
                    actual_width <
                    0.65 * w
                ):
                    lane_width = (
                        0.80 * lane_width +
                        0.20 * actual_width
                    )
                    _prev_lane_width = lane_width
 
                center_x = (
                    white_x +
                    yellow_x
                ) / 2.0
 
            # Yellow temporarily missing
            else:
 
                center_x = (
                    white_x +
                    lane_width / 2.0
                )
 
    # ---------------------------------------------------------
    # Dashed white line may disappear temporarily.
    # Use previous valid result rather than immediately giving up.
    # ---------------------------------------------------------
    if lane == "unknown":
 
        if (
            _prev_lane != "unknown" and
            _prev_center_x >= 0
        ):
            return {
                "center_x": int(_prev_center_x),
                "lane": _prev_lane
            }
 
        return {
            "center_x": -1,
            "lane": "unknown"
        }
 
    # ---------------------------------------------------------
    # Validate centre
    # ---------------------------------------------------------
    if not np.isfinite(center_x):
        return {
            "center_x": -1,
            "lane": "unknown"
        }
 
    # ---------------------------------------------------------
    # Smooth only moderate frame-to-frame changes.
    # This reduces isolated noisy detections while still allowing
    # the centre to move through a genuine curve.
    # ---------------------------------------------------------
    if (
        _prev_lane == lane and
        _prev_center_x >= 0
    ):
 
        difference = abs(
            center_x -
            _prev_center_x
        )
 
        if difference < 0.15 * w:
 
            center_x = (
                0.65 * center_x +
                0.35 * _prev_center_x
            )
 
        else:
 
            center_x = (
                0.30 * center_x +
                0.70 * _prev_center_x
            )
 
    center_x = int(
        round(center_x)
    )
 
    # Final bounds check
    if not (0 <= center_x < w):
        return {
            "center_x": -1,
            "lane": "unknown"
        }
 
    # Save for next frame
    _prev_lane = lane
    _prev_center_x = center_x
 
    return {
        "center_x": center_x,
        "lane": lane
    }
 
 
def _fit_lane_curve(x, y):
    """
    Robust quadratic lane fit:
 
        x = a*y^2 + b*y + c
 
    Outlier pixels are repeatedly rejected.
    """
 
    if len(x) < 15:
        return None
 
    x = np.asarray(
        x,
        dtype=np.float64
    )
 
    y = np.asarray(
        y,
        dtype=np.float64
    )
 
    valid = (
        np.isfinite(x) &
        np.isfinite(y)
    )
 
    x = x[valid]
    y = y[valid]
 
    if len(x) < 15:
        return None
 
    if np.ptp(y) < 30:
        return None
 
    try:
        coeff = np.polyfit(
            y,
            x,
            2
        )
    except (np.linalg.LinAlgError, ValueError):
        return None
 
    # Robust iterative fitting
    for _ in range(5):
 
        predicted = np.polyval(
            coeff,
            y
        )
 
        residual = np.abs(
            x - predicted
        )
 
        median_error = np.median(
            residual
        )
 
        threshold = max(
            7.0,
            2.5 * median_error
        )
 
        inliers = residual <= threshold
 
        if np.count_nonzero(inliers) < 10:
            break
 
        try:
            new_coeff = np.polyfit(
                y[inliers],
                x[inliers],
                2
            )
        except (np.linalg.LinAlgError, ValueError):
            break
 
        coeff = new_coeff
 
    return coeff
 
 
################ END OF YOUR IMPLEMENTATION ##################

def detect_lane(frame):
    '''
    Purpose:
    ---
    Detect the lane in a single frame and report where the centre of the lane
    is, and which of the two lanes the vehicle is currently in.

    Input Arguments:
    ---
    `frame` :   [ numpy.ndarray ]
        A single BGR frame read from the video, of shape (height, width, 3).

    Returns:
    ---
    `result` :  [ dict ]
        {
            "center_x" : int,   x-pixel of the lane centre in this frame,
                                or -1 if the lane could not be found
            "lane"     : str,   "left", "right" or "unknown"
        }

    Example call:
    ---
    result = detect_lane(frame)

    COORDINATE SYSTEM:
    ---
    `center_x` is an absolute pixel column in the frame AS RECEIVED - the
    dataset's own resolution, 640x480. It is compared against a ground truth
    measured in those pixels, so it only means anything in them.

    You may resize, crop or warp all you like inside this function, but scale
    the answer back before returning it. A centre found in a 320x240 copy is
    half the value it should be, and a centre read off a bird's-eye view is in
    warped coordinates, not frame ones - map the point back through the inverse
    of your transform. Do not re-encode or resize the clip files themselves.

    NOTE:
    ---
    This function must ONLY compute and return the result.
    Do not call cv2.imshow(), cv2.waitKey(), cv2.imwrite() or print() from
    inside it. All visualisation and debugging output belongs outside this
    function - see draw_overlay() and process_video() below.
    '''

    center_x = -1
    lane = LANE_UNKNOWN

    #################### ADD YOUR CODE HERE ####################
    # 1. Isolate the lane markings in `frame`
    # 2. Work out which two markings bracket the vehicle
    # 3. Compute the x-pixel of the lane centre   ->  center_x
    # 4. Decide which lane the vehicle is in      ->  lane
    ############################################################

    return {"center_x": center_x, "lane": lane}


# ------------------------------------------------------------------
# Add any helper functions and global variables you need below this
# comment, and keep them ABOVE the "END OF YOUR IMPLEMENTATION" line.
# They must be called from detect_lane() - the evaluation script only
# ever calls that one function. List them in the file header too.
# ------------------------------------------------------------------


##############################################################
################ END OF YOUR IMPLEMENTATION ##################
##############################################################


#################### DO NOT EDIT BELOW THIS LINE ####################

def validate_result(result, frame_index):
    '''
    Purpose:
    ---
    Check that detect_lane() returned the expected structure and normalise it,
    so that a malformed return is reported here instead of silently scoring
    zero during evaluation.

    Input Arguments:
    ---
    `result` :          [ object ]      whatever detect_lane() returned
    `frame_index` :     [ int ]         index of the frame, used in error messages

    Returns:
    ---
    `clean` :           [ dict ]        {"center_x": int, "lane": str}
    '''
    where = "detect_lane() on frame {}".format(frame_index)

    if not isinstance(result, dict):
        raise TypeError("{} must return a dict, got {}".format(where, type(result).__name__))

    missing = {"center_x", "lane"} - set(result.keys())
    if missing:
        raise ValueError("{} is missing the key(s): {}".format(where, ", ".join(sorted(missing))))

    center_x = result["center_x"]
    if isinstance(center_x, bool) or not isinstance(center_x, (int, float, np.integer, np.floating)):
        raise TypeError("{} returned center_x of type {}, expected a number".format(
            where, type(center_x).__name__))
    center_x = int(round(float(center_x)))

    lane = result["lane"]
    if not isinstance(lane, str):
        raise TypeError("{} returned lane of type {}, expected a string".format(
            where, type(lane).__name__))
    lane = lane.strip().lower()
    if lane not in VALID_LANES:
        raise ValueError("{} returned lane = '{}', expected one of {}".format(
            where, result["lane"], ", ".join(VALID_LANES)))

    return {"center_x": center_x, "lane": lane}


def draw_overlay(frame, result):
    '''
    Purpose:
    ---
    Draw the detected lane centre and lane label on a copy of the frame.
    This is where display code belongs - never inside detect_lane().

    Input Arguments:
    ---
    `frame` :   [ numpy.ndarray ]   the frame that was passed to detect_lane()
    `result` :  [ dict ]            the validated result for that frame

    Returns:
    ---
    `canvas` :  [ numpy.ndarray ]   a copy of the frame with the overlay drawn
    '''
    canvas = frame.copy()
    height, width = canvas.shape[:2]

    # frame centre, for reference - roughly where the vehicle is pointing
    cv2.line(canvas, (width // 2, height), (width // 2, height - 40), (128, 128, 128), 1)

    center_x = result["center_x"]
    if 0 <= center_x < width:
        cv2.line(canvas, (center_x, height), (center_x, height // 2), (0, 0, 255), 2)
        cv2.circle(canvas, (center_x, height - 10), 5, (0, 0, 255), -1)

    label = "lane: {}   center_x: {}".format(result["lane"], center_x)
    cv2.putText(canvas, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return canvas


def process_video(video_path, show=False):
    '''
    Purpose:
    ---
    Read a video frame by frame, hand each frame to detect_lane() and collect
    the results.

    Input Arguments:
    ---
    `video_path` :  [ str ]     path to the video file
    `show` :        [ bool ]    if True, display the overlay while processing

    Returns:
    ---
    `results` :     [ list ]    one dict per frame:
                                {"frame": int, "center_x": int, "lane": str}
    '''
    if not os.path.isfile(video_path):
        raise FileNotFoundError("no such video file: {}".format(video_path))

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise IOError("OpenCV could not open the video: {}".format(video_path))

    window = "Task 1C - {}".format(os.path.basename(video_path))
    results = []
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            # A copy is passed in, so anything drawn inside detect_lane() cannot
            # corrupt the frame used for display.
            result = validate_result(detect_lane(frame.copy()), frame_index)
            results.append({"frame": frame_index, **result})

            if show:
                cv2.imshow(window, draw_overlay(frame, result))
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break

            frame_index += 1
    finally:
        capture.release()
        if show:
            cv2.destroyAllWindows()

    if not results:
        raise IOError("no frames could be read from: {}".format(video_path))

    return results


def summarise(video_path, results):
    '''
    Purpose:
    ---
    Print a one-line-per-video summary, so you can see at a glance whether the
    detector is returning anything sensible.

    Input Arguments:
    ---
    `video_path` :  [ str ]     path to the video that was processed
    `results` :     [ list ]    output of process_video()

    Returns:
    ---
    None
    '''
    total = len(results)
    counts = {lane: 0 for lane in VALID_LANES}
    for entry in results:
        counts[entry["lane"]] += 1
    not_found = sum(1 for entry in results if entry["center_x"] < 0)

    print("{:<28} {:>5} frames | left {:>5} | right {:>5} | unknown {:>5} | no centre {:>5}".format(
        os.path.basename(video_path), total,
        counts[LANE_LEFT], counts[LANE_RIGHT], counts[LANE_UNKNOWN], not_found))


def expand_videos(paths):
    '''
    Purpose:
    ---
    Turn the command-line arguments into a list of video files, accepting a
    FOLDER as well as individual files.

    A folder is the portable way to say "all the clips": Windows shells do not
    expand `public/*.mp4` the way bash does - cmd and PowerShell hand the
    pattern through verbatim and the script would look for a file literally
    named "*.mp4". `python lane_detection.py public` behaves the same on every
    platform.

    Input Arguments:
    ---
    `paths` :   [ list ]    the raw command-line arguments

    Returns:
    ---
    `videos` :  [ list ]    paths to individual video files, folders expanded
    '''
    videos = []
    for raw in paths:
        if os.path.isdir(raw):
            found = sorted(f for f in os.listdir(raw) if f.lower().endswith(".mp4"))
            if not found:
                raise FileNotFoundError("no .mp4 files in the folder: {}".format(raw))
            videos.extend(os.path.join(raw, f) for f in found)
        else:
            videos.append(raw)
    return videos


def main():
    parser = argparse.ArgumentParser(
        description="Task 1C - run your lane detector over one or more videos.")
    parser.add_argument("videos", nargs="+",
                        help="video file(s), or a folder holding them "
                             "(e.g. 'public')")
    parser.add_argument("--show", action="store_true",
                        help="display the detection overlay while processing (press q to stop)")
    parser.add_argument("--out", metavar="FILE",
                        help="write the per-frame results to this JSON file")
    args = parser.parse_args()

    all_results = {}
    for video_path in expand_videos(args.videos):
        results = process_video(video_path, show=args.show)
        summarise(video_path, results)
        all_results[os.path.basename(video_path)] = results

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(all_results, handle, indent=2)
        print("\nresults written to {}".format(args.out))


if __name__ == "__main__":
    main()
