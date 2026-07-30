import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import RRDBNet_arch as arch
# Paths
model_path = 'models/RRDB_ESRGAN_x4.pth'
background_path = 'assets/background.png'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load ESRGAN model
model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
model.eval()
model = model.to(device)

# ---------------- METRICS ----------------

def mse(img1,img2):
    return np.mean((img1-img2)**2)

def rmse(img1,img2):
    return math.sqrt(mse(img1,img2))

def psnr(img1,img2):
    return cv2.PSNR(img1,img2)

def snr(img1,img2):
    signal=np.mean(img1**2)
    noise=np.mean((img1-img2)**2)
    return 10*np.log10(signal/noise)

def entropy(img):
    hist=cv2.calcHist([img],[0],None,[256],[0,256])
    hist=hist.ravel()/hist.sum()
    return -np.sum(hist*np.log2(hist+1e-7))


# ---------------- APPLICATION ----------------

class ESRGAN_APP:

    def __init__(self,root):
        

        self.root=root
        self.root.title("ESRGAN Cinematic Enhancer")
        self.root.geometry("1300x750")

        self.original=None
        self.enhanced=None
        self.last_filter_used="Enhance"

        # ---------- BACKGROUND ----------
        bg=Image.open("assets/background.png")
        bg=bg.resize((1300,750))
        self.bg_img=ImageTk.PhotoImage(bg)

        bg_label=tk.Label(root,image=self.bg_img)
        bg_label.place(x=0,y=0,relwidth=1,relheight=1)

        # ---------- TITLE ----------
        title=tk.Label(root,text="✨ ESRGAN Super-Resolution",
                       font=("Arial",32,"bold"),
                       fg="white",bg="#000000")
        title.pack(pady=10)

        # ---------- IMAGE FRAME ----------
        frame=tk.Frame(root,bg="#000000")
        frame.pack(pady=20)

        self.left_label=tk.Label(frame,bg="black")
        self.left_label.grid(row=0,column=0,padx=40)

        self.right_label=tk.Label(frame,bg="black")
        self.right_label.grid(row=0,column=1,padx=40)

        # ---------- BUTTONS ----------
        btn=tk.Frame(root,bg="#000000")
        btn.pack(pady=30)

        # -------- FIRST ROW --------
        tk.Button(btn,text="Upload",bg="#2e86de",fg="white",
                width=12,font=("Arial", 11, "bold"),command=self.upload).grid(row=0,column=0,padx=10,pady=5)

        tk.Button(btn,text="Quality Check",bg="#2c3e50",fg="white",
                width=12,font=("Arial", 11, "bold"),command=self.quality_assessment).grid(row=0,column=1,padx=10,pady=5)

        tk.Button(btn,text="Enhance",bg="#27ae60",fg="white",
                width=12,font=("Arial", 11, "bold"),command=self.enhance).grid(row=0,column=2,padx=10,pady=5)

        tk.Button(btn,text="Deblur",bg="#17a589",fg="white",
                width=12,font=("Arial", 11, "bold"),command=self.deblur).grid(row=0,column=3,padx=10,pady=5)

        tk.Button(btn,text="High Pass",bg="#8e44ad",fg="white",
                width=12,font=("Arial", 11, "bold"),command=self.highpass).grid(row=0,column=4,padx=10,pady=5)

        tk.Button(btn,text="Low Pass",bg="#16a085",fg="white",
                width=12,font=("Arial", 11, "bold"),command=self.lowpass).grid(row=0,column=5,padx=10,pady=5)


        # -------- SECOND ROW --------
        tk.Button(btn,text="Upscale",bg="#e67e22",fg="white",
                width=12,font=("Arial", 11, "bold"),command=self.upscale).grid(row=1,column=0,padx=10,pady=5)

        tk.Button(btn,text="Compare",bg="#7d3c98",fg="white",
                width=12,font=("Arial", 11, "bold"),command=self.compare).grid(row=1,column=1,padx=10,pady=5)

        tk.Button(btn,text="Auto Enhance",bg="#c0392b",fg="white",
                width=12,font=("Arial", 11, "bold"),command=self.auto_enhance).grid(row=1,column=2,padx=10,pady=5)

        tk.Button(btn,text="Download",bg="#f1c40f",
                width=12,font=("Arial", 11, "bold"),command=self.download).grid(row=1,column=3,padx=10,pady=5)

        tk.Button(btn,text="Video Enhance",bg="#d35400",fg="white",
                width=14,font=("Arial", 11, "bold"),command=self.video_enhance).grid(row=1,column=4,padx=20,pady=5)
        
        tk.Button(btn,text="Night Enhance",
          bg="#34495e",fg="white",
          width=12,font=("Arial", 11, "bold"),command=self.low_light_enhance).grid(row=1,column=5,padx=10,pady=5)
        
    
                

    # ---------- UPLOAD ----------
    def upload(self):

        file=filedialog.askopenfilename()

        if file:
            img=cv2.imread(file)
            self.original=img

            show=cv2.resize(img,(400,350))
            show=cv2.cvtColor(show,cv2.COLOR_BGR2RGB)

            im=ImageTk.PhotoImage(Image.fromarray(show))
            self.left_label.configure(image=im)
            self.left_label.image=im

    # ---------- DISPLAY ----------
    def show_enhanced(self):

        show=cv2.resize(self.enhanced,(400,350))
        show=cv2.cvtColor(show,cv2.COLOR_BGR2RGB)

        im=ImageTk.PhotoImage(Image.fromarray(show))
        self.right_label.configure(image=im)
        self.right_label.image=im

    # ---------- FILTERS ----------
    """def enhance(self):
        self.enhanced=cv2.detailEnhance(self.original,sigma_s=10,sigma_r=0.15)
        self.last_filter_used="Enhanced Image"
        self.show_enhanced()"""
    
    def enhance(self):
        if self.original is None:
            return

        img = self.original.astype(np.float32) / 255.0
        img = torch.from_numpy(np.transpose(img[:, :, [2,1,0]], (2,0,1))).unsqueeze(0).float().to(device)

        with torch.no_grad():
            output = model(img).data.squeeze().cpu().clamp_(0,1).numpy()

        output = np.transpose(output[[2,1,0], :, :], (1,2,0))
        output = (output * 255.0).astype(np.uint8)

        self.enhanced = output
        self.last_filter_used = "Enhanced Image"
        self.show_enhanced()


    def deblur(self):
        if self.original is None: return
        kernel=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        self.enhanced=cv2.filter2D(self.original,-1,kernel)
        self.last_filter_used="Deblurred Image"
        self.show_enhanced()

    def highpass(self):
        if self.original is None: return
        kernel=np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
        self.enhanced=cv2.filter2D(self.original,-1,kernel)
        self.last_filter_used="High Pass Filtered Image"
        self.show_enhanced()

    def lowpass(self):
        if self.original is None: return
        self.enhanced=cv2.GaussianBlur(self.original,(7,7),0)
        self.last_filter_used="Low Pass Filtered Image"
        self.show_enhanced()

    def upscale(self):
        if self.original is None: return
        self.enhanced=cv2.resize(self.original,None,fx=2,fy=2,
                                 interpolation=cv2.INTER_CUBIC)
        self.last_filter_used="Upscaled Image"
        self.show_enhanced()

    # ---------- DOWNLOAD ----------
    def download(self):
        if self.enhanced is None: return
        path=filedialog.asksaveasfilename(defaultextension=".png")
        if path:
            cv2.imwrite(path,self.enhanced)

    # ---------- VIDEO ----------
    """def video_enhance(self):

        file=filedialog.askopenfilename()
        cap=cv2.VideoCapture(file)

        while True:

            ret,frame=cap.read()
            if not ret:
                break

            enhanced=cv2.detailEnhance(frame,sigma_s=10,sigma_r=0.15)

            frame=cv2.resize(frame,(500,350))
            enhanced=cv2.resize(enhanced,(500,350))

            combined=np.hstack((frame,enhanced))

            cv2.putText(combined,"Original",(180,30),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

            cv2.putText(combined,"Enhanced",(700,30),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

            cv2.imshow("Video Enhancement",combined)

            if cv2.waitKey(1)==27:
                break

        cap.release()
        cv2.destroyAllWindows()"""


    def video_enhance(self):

        file = filedialog.askopenfilename()
        if not file:
            return

        cap = cv2.VideoCapture(file)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30

        delay = int(1000 / fps)

        # -------- METRIC STORAGE --------
        mse_list, rmse_list, psnr_list = [], [], []
        snr_list, ssim_list, entropy_list = [], [], []

        frame_count = 0

        while True:

            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # -------- Resize --------
            original = cv2.resize(frame, (500,350))

            # ---------- ENHANCEMENT ----------
            blur = cv2.GaussianBlur(frame,(3,3),0)

            lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
            l,a,b = cv2.split(lab)

            clahe = cv2.createCLAHE(clipLimit=1.8,tileGridSize=(8,8))
            l = clahe.apply(l)

            lab = cv2.merge((l,a,b))
            contrast = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            gaussian = cv2.GaussianBlur(contrast,(0,0),1.5)
            enhanced = cv2.addWeighted(contrast,1.7,gaussian,-0.7,0)

            enhanced = cv2.resize(enhanced,(500,350))

            # ---------- METRICS (EVERY 5th FRAME FOR SPEED) ----------
            if frame_count % 5 == 0:

                g1 = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
                g2 = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

                g2 = cv2.resize(g2,(g1.shape[1], g1.shape[0]))

                mse_list.append(mse(g1,g2))
                rmse_list.append(rmse(g1,g2))
                psnr_list.append(psnr(g1,g2))
                snr_list.append(snr(g1,g2))
                ssim_list.append(ssim(g1,g2))
                entropy_list.append(entropy(g2))

            # ---------- DISPLAY ----------
            combined = np.hstack((original, enhanced))

            cv2.putText(combined,"Original",(170,30),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

            cv2.putText(combined,"Enhanced",(650,30),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

            cv2.imshow("Video Enhancement", combined)

            if cv2.waitKey(delay) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

        # ---------- SAFE AVERAGE ----------
        def avg(lst):
            return sum(lst)/len(lst) if len(lst) > 0 else 0

        self.video_metrics_result = {
            "SNR": avg(snr_list),
            "PSNR": avg(psnr_list),
            "SSIM": avg(ssim_list),
            "MSE": avg(mse_list),
            "RMSE": avg(rmse_list),
            "Entropy": avg(entropy_list)
        }

        self.show_video_metrics()

    def show_video_metrics(self):

        win = tk.Toplevel(self.root)
        win.title("Video Metrics")
        win.geometry("400x400")
        win.configure(bg="black")

        tk.Label(win, text="Video Quality Metrics",
                font=("Arial",16,"bold"),
                bg="black", fg="white").pack(pady=15)

        for key, value in self.video_metrics_result.items():
            tk.Label(win,
                    text=f"{key}: {value:.4f}",
                    bg="gray30", fg="white",
                    font=("Arial",12), width=30).pack(pady=5)

        tk.Button(win,
                text="Show Graph",
                bg="#9b59b6", fg="white",
                font=("Arial",11,"bold"),
                command=self.plot_video_overall_metrics).pack(pady=15)
        tk.Button(win,
          text="Radar Chart",
          bg="#e67e22", fg="white",
          font=("Arial",11,"bold"),
          command=self.plot_video_radar_chart).pack(pady=10)
            
    def plot_video_overall_metrics(self):

        metrics = self.video_metrics_result

        plt.figure(figsize=(7,5))

        names = list(metrics.keys())
        values = list(metrics.values())

        plt.bar(names, values)

        plt.title("Overall Video Metrics")
        plt.xlabel("Metrics")
        plt.ylabel("Values")

        plt.grid(True)

        plt.show()

    def plot_video_radar_chart(self):

        metrics = self.video_metrics_result

        labels = list(metrics.keys())
        values = list(metrics.values())

        # ---------- NORMALIZATION ----------
        values = np.array(values, dtype=np.float32)

        # Avoid division by zero
        values = values / (np.max(values) + 1e-8)

        # Close the loop
        values = np.append(values, values[0])
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        angles = np.append(angles, angles[0])

        # ---------- PLOT ----------
        plt.figure(figsize=(6,6))
        ax = plt.subplot(111, polar=True)

        ax.plot(angles, values)
        ax.fill(angles, values, alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)

        plt.title("Video Metrics Radar Chart")

        plt.show()

    """def video_enhance(self):

        file = filedialog.askopenfilename()
        if not file:
            return

        cap = cv2.VideoCapture(file)

        # Get video FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30  # fallback if FPS not detected

        delay = int(1000 / fps)  # delay in milliseconds

        while True:

            ret, frame = cap.read()
            if not ret:
                break

            original = cv2.resize(frame,(500,350))

            # ---------- DENOISE ----------
            denoise = cv2.bilateralFilter(frame,5,50,50)

            # ---------- CONTRAST ----------
            alpha = 1.2   # contrast
            beta = -15     # brightness
            contrast = cv2.convertScaleAbs(denoise, alpha=alpha, beta=beta)

            # ---------- SHARPEN ----------
            kernel = np.array([[0,-1,0],
                            [-1,5,-1],
                            [0,-1,0]])

            sharpen = cv2.filter2D(contrast,-1,kernel)

            enhanced = cv2.resize(sharpen,(500,350))

            combined = np.hstack((original, enhanced))

            cv2.putText(combined,"Original",(170,30),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

            cv2.putText(combined,"Enhanced",(650,30),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

            cv2.imshow("Video Enhancement", combined)

            # Use FPS-based delay for normal speed
            if cv2.waitKey(delay) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()"""
    
    """def video_enhance(self):

        file = filedialog.askopenfilename()
        if not file:
            return

        cap = cv2.VideoCapture(file)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30

        delay = int(1000 / fps)

        while True:

            ret, frame = cap.read()
            if not ret:
                break

            original = cv2.resize(frame,(500,350))

                        # ---------- DENOISE ----------
            denoise = cv2.GaussianBlur(frame,(5,5),0)

            # ---------- CONTRAST ----------
            alpha = 1.1
            beta = 10
            contrast = cv2.convertScaleAbs(denoise, alpha=alpha, beta=beta)

            # ---------- SHARPEN ----------
            kernel = np.array([[0,-1,0],
                            [-1,5,-1],
                            [0,-1,0]])

            sharpen = cv2.filter2D(contrast,-1,kernel)

            enhanced = cv2.resize(sharpen,(500,350))

            combined = np.hstack((original, enhanced))

            cv2.putText(combined,"Original",(170,30),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

            cv2.putText(combined,"Enhanced",(650,30),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

            cv2.imshow("Video Enhancement", combined)

            if cv2.waitKey(delay) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()"""
    
    def quality_assessment(self):
        if self.original is None: return
        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)

        # ---- Quality Metrics ----
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        contrast = np.std(gray)
        brightness = np.mean(gray)
        noise = np.var(gray)

        # ---- Create GUI Window ----
        win = tk.Toplevel(self.root)
        win.title("Image Quality Assessment")
        win.geometry("400x350")
        win.configure(bg="black")

        tk.Label(win,text="Image Quality Analysis",
                font=("Arial",16,"bold"),
                bg="black",fg="white").pack(pady=15)

        tk.Label(win,text=f"Blur Score : {blur_score:.2f}",
                font=("Arial",12),
                bg="gray20",fg="white",width=30).pack(pady=8)

        tk.Label(win,text=f"Contrast : {contrast:.2f}",
                font=("Arial",12),
                bg="gray30",fg="white",width=30).pack(pady=8)

        tk.Label(win,text=f"Brightness : {brightness:.2f}",
                font=("Arial",12),
                bg="gray40",fg="white",width=30).pack(pady=8)

        tk.Label(win,text=f"Noise Level : {noise:.2f}",
                font=("Arial",12),
                bg="gray50",fg="white",width=30).pack(pady=8)

        # ---- Quality Interpretation ----
        if blur_score < 80:
            quality = "Blurry Image Detected"
        elif contrast < 40:
            quality = "Low Contrast Image"
        elif noise > 500:
            quality = "High Noise Detected"
        elif blur_score>80 and blur_score<120:
            quality="Slightly Blur Detected"
        else:
            quality = "Good Quality Image"

        tk.Label(win,text=f"Result : {quality}",
                font=("Arial",12,"bold"),
                bg="#1abc9c",fg="black",width=30).pack(pady=15)

    def auto_enhance(self):
        if self.original is None: return
        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)

        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        contrast = np.std(gray)
        noise = np.var(gray)

        # ---- Decision System ----
        if blur < 80:
            problem = "Blur Detected"
            action = "Deblur Filter Applied"
            self.deblur()

        elif contrast < 40:
            problem = "Low Contrast Detected"
            action = "Detail Enhance Applied"
            self.enhance()

        elif noise > 500:
            problem = "High Noise Detected"
            action = "Low Pass Filter Applied"
            self.lowpass()

        elif blur < 120 and blur>80:
            problem = "Slight Blur Detected"
            action = "High Pass Filter Applied"
            self.highpass()

        else:
            problem = "No Major Issue"
            action = "Detail Enhance Applied"
            self.enhance()

        # ---- GUI Result Window ----
        win = tk.Toplevel(self.root)
        win.title("Auto Enhancement Decision")
        win.geometry("420x300")
        win.configure(bg="black")

        tk.Label(win,text="Automatic Enhancement System",
                font=("Arial",16,"bold"),
                bg="black",fg="white").pack(pady=20)

        tk.Label(win,text=f"Blur Score : {blur:.2f}",
                bg="gray20",fg="white",
                font=("Arial",11),width=30).pack(pady=5)

        tk.Label(win,text=f"Contrast : {contrast:.2f}",
                bg="gray30",fg="white",
                font=("Arial",11),width=30).pack(pady=5)

        tk.Label(win,text=f"Noise Level : {noise:.2f}",
                bg="gray40",fg="white",
                font=("Arial",11),width=30).pack(pady=5)

        tk.Label(win,text=f"Detected Issue : {problem}",
                bg="#e74c3c",fg="white",
                font=("Arial",12,"bold"),width=30).pack(pady=10)

        tk.Label(win,text=f"Action Taken : {action}",
                bg="#2ecc71",fg="black",
                font=("Arial",12,"bold"),width=30).pack(pady=10)
        
    def low_light_enhance(self):

        if self.original is None:
            return

        img = self.original.copy()

        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        l,a,b = cv2.split(lab)

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        l = clahe.apply(l)

        lab = cv2.merge((l,a,b))

        result = cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)

        self.enhanced = result
        self.last_filter_used="Night Enhancement"

        self.show_enhanced()

    # ---------- GRAPHS ----------
    def plot_snr_psnr(self):

        plt.figure(figsize=(5,4))
        plt.title(f"{self.last_filter_used} - SNR & PSNR")

        plt.bar(["SNR","PSNR"],
                [self.snr_value,self.psnr_value],
                color=["#007bff","#28a745"])

        plt.grid(True)
        plt.show()

    def plot_mse_rmse(self):

        plt.figure(figsize=(5,4))
        plt.title(f"{self.last_filter_used} - MSE & RMSE")

        plt.bar(["MSE","RMSE"],
                [self.mse_value,self.rmse_value],
                color=["#ffc107","#6c757d"])

        plt.grid(True)
        plt.show()

    def plot_ssim_entropy(self):

        plt.figure(figsize=(5,4))
        plt.title(f"{self.last_filter_used} - SSIM & Entropy")

        plt.bar(["SSIM","Entropy"],
                [self.ssim_value,self.entropy_value],
                color=["#6610f2","#20c997"])

        plt.grid(True)
        plt.show()

    def plot_all_metrics(self):

        metrics={
            "SNR":self.snr_value,
            "PSNR":self.psnr_value,
            "SSIM":self.ssim_value,
            "MSE":self.mse_value,
            "RMSE":self.rmse_value,
            "Entropy":self.entropy_value
        }

        plt.figure(figsize=(8,5))

        plt.plot(list(metrics.keys()),
                 list(metrics.values()),
                 marker='o',
                 color='cyan')

        plt.title(f"{self.last_filter_used} - All Metrics")
        plt.grid(True)
        plt.show()

    # ---------- COMPARE WINDOW ----------
    def compare(self):
        if self.original is None or self.enhanced is None: return
        win=tk.Toplevel(self.root)

        # Dynamic title
        win.title(f"Comparison: Original vs {self.last_filter_used}")

        win.geometry("1000x650")
        win.configure(bg="black")

        o=cv2.resize(self.original,(350,300))
        e=cv2.resize(self.enhanced,(350,300))

        o=cv2.cvtColor(o,cv2.COLOR_BGR2RGB)
        e=cv2.cvtColor(e,cv2.COLOR_BGR2RGB)

        im1=ImageTk.PhotoImage(Image.fromarray(o))
        im2=ImageTk.PhotoImage(Image.fromarray(e))

        tk.Label(win,text="Original Image\nBefore",
                 fg="white",bg="black",
                 font=("Arial",14,"bold")).place(x=200,y=20)

        tk.Label(win,text=f"{self.last_filter_used} \nAfter",
                 fg="white",bg="black",
                 font=("Arial",14,"bold")).place(x=650,y=20)

        l1=tk.Label(win,image=im1)
        l1.image=im1
        l1.place(x=80,y=80)

        l2=tk.Label(win,image=im2)
        l2.image=im2
        l2.place(x=550,y=80)

        g1=cv2.cvtColor(self.original,cv2.COLOR_BGR2GRAY)
        g2=cv2.cvtColor(self.enhanced,cv2.COLOR_BGR2GRAY)

        # resize enhanced for metric calculation (important for Upscale)
        g2=cv2.resize(g2,(g1.shape[1],g1.shape[0]))

        self.mse_value=mse(g1,g2)
        self.rmse_value=rmse(g1,g2)
        self.psnr_value=psnr(g1,g2)
        self.snr_value=snr(g1,g2)
        self.ssim_value=ssim(g1,g2)
        self.entropy_value=entropy(g2)

        tk.Label(win,text=f"SNR: {self.snr_value:.4f}",
                 bg="gray30",fg="white",width=20).place(x=150,y=420)

        tk.Label(win,text=f"PSNR: {self.psnr_value:.4f}",
                 bg="#1e8449",fg="white",width=20).place(x=600,y=420)

        tk.Label(win,text=f"SSIM: {self.ssim_value:.4f}",
                 bg="#1f618d",fg="white",width=20).place(x=150,y=460)

        tk.Label(win,text=f"MSE: {self.mse_value:.4f}",
                 bg="#b7950b",fg="white",width=20).place(x=600,y=460)

        tk.Label(win,text=f"RMSE: {self.rmse_value:.4f}",
                 bg="gray50",fg="white",width=20).place(x=150,y=500)

        tk.Label(win,text=f"Entropy: {self.entropy_value:.4f}",
                 bg="#117a65",fg="white",width=20).place(x=600,y=500)

        tk.Button(win,text="View SNR & PSNR",
                  width=18,command=self.plot_snr_psnr).place(x=120,y=560)

        tk.Button(win,text="View MSE & RMSE",
                  width=18,command=self.plot_mse_rmse).place(x=360,y=560)

        tk.Button(win,text="View SSIM & Entropy",
                  width=18,command=self.plot_ssim_entropy).place(x=600,y=560)

        tk.Button(win,text="View All Metrics",
                  width=18,command=self.plot_all_metrics).place(x=820,y=560)


root=tk.Tk()
app=ESRGAN_APP(root)
root.mainloop()
