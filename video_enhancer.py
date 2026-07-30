import cv2
import torch
import numpy as np
import RRDBNet_arch as arch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = "models/RRDB_ESRGAN_x4.pth"

model = arch.RRDBNet(3,3,64,23,gc=32)
model.load_state_dict(torch.load(model_path,map_location=device))
model.eval()
model=model.to(device)

def enhance_frame(frame):

    img=frame.astype(np.float32)/255.0

    img=torch.from_numpy(
        np.transpose(img[:,:, [2,1,0]],(2,0,1))
    ).unsqueeze(0).float().to(device)

    with torch.no_grad():
        output=model(img).data.squeeze().float().cpu().clamp_(0,1).numpy()

    output=np.transpose(output[[2,1,0],:,:],(1,2,0))
    output=(output*255.0).astype(np.uint8)

    return output


def enhance_video(video_path, output_path):

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width * 4, height * 4)
    )

    frame_count = 0
    print(f"Enhancing video device: {device} | Total frames: {total_frames}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            enhanced_frame = enhance_frame(frame)
            out.write(enhanced_frame)

            if total_frames > 0:
                pct = (frame_count / total_frames) * 100
                print(f"Processing frame {frame_count}/{total_frames} ({pct:.1f}%)...", end="\r", flush=True)
            else:
                print(f"Processing frame {frame_count}...", end="\r", flush=True)
    except KeyboardInterrupt:
        print(f"\nProcessing interrupted by user! Finalizing video up to frame {frame_count}...")
    finally:
        cap.release()
        out.release()

    print(f"Video saved! Total frames written: {frame_count}.")


if __name__ == '__main__':
    import os
    import tkinter as tk
    from tkinter import filedialog

    # Hide Tkinter root window
    root = tk.Tk()
    root.withdraw()

    # Ensure model path works whether run from root or ESRGAN subfolder
    if not os.path.exists(model_path) and os.path.exists("ESRGAN/" + model_path):
        model_path = "ESRGAN/" + model_path
        model.load_state_dict(torch.load(model_path, map_location=device))

    print("Please select an input video file...")
    video_input = filedialog.askopenfilename(
        title="Select Input Video",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
    )

    if video_input:
        print(f"Selected input video: {video_input}")
        output_suggested = os.path.splitext(video_input)[0] + "_enhanced.mp4"
        video_output = filedialog.asksaveasfilename(
            title="Save Enhanced Video As",
            initialfile=os.path.basename(output_suggested),
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4")]
        )

        if video_output:
            print(f"Processing video to: {video_output} ...")
            enhance_video(video_input, video_output)
        else:
            print("No output path specified. Cancelled.")
    else:
        print("No video selected. Cancelled.")