import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import soundfile as sf
import sounddevice as sd
import numpy as np
import threading
import librosa
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
try:
    import pacmap
    PACMAP_AVAILABLE = True
except ImportError:
    PACMAP_AVAILABLE = False
    print("Warning: PaCMAP not installed. Install with: pip install pacmap")
import os
import time

MIN_SENSITIVITY = 1
MAX_SENSITIVITY = 20

class TimbreReorganizerPCA:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Timbre Reorganizer (PCA + MFCC)")

        self.audio = None
        self.sorted_audio_pca = None
        self.sorted_audio_pacmap = None
        self.sr = None
        self.last_boundaries = None
        self.pca_scores = None
        self.pacmap_scores = None
        self.pca_chunks = None
        self.pacmap_chunks = None
        
        # Variables to track selected points
        self.selected_index = None
        self.selected_method = None
        self.current_scatter_plots = {}  # Store scatter plot references
        
        # NEW: Audio buffers for instant playback
        self.pca_audio_buffers = []  # List of numpy arrays
        self.pacmap_audio_buffers = []
        self.current_playback_thread = None
        self.is_playing = False

        self.order_var = tk.StringVar(value="Similar → Different")
        self.segmentation_var = tk.StringVar(value="Onset Detection")
        self.method_var = tk.StringVar(value="PCA")

        # GUI Buttons and Controls
        tk.Button(root, text="Upload Audio", width=45, command=self.load_audio).pack(pady=4)
        tk.Button(root, text="Play Original Audio", width=45, command=self.play_original).pack(pady=4)

        order_frame = tk.Frame(root)
        order_frame.pack(pady=4)
        tk.Label(order_frame, text="Order:").pack(side="left", padx=5)
        tk.OptionMenu(order_frame, self.order_var,
                      "Similar → Different", "Different → Similar").pack(side="left")

        seg_frame = tk.Frame(root)
        seg_frame.pack(pady=4)
        tk.Label(seg_frame, text="Segmentation:").pack(side="left", padx=5)
        tk.OptionMenu(seg_frame, self.segmentation_var,
                      "Onset Detection", "Novelty Detection", "Combined").pack(side="left")

        method_frame = tk.Frame(root)
        method_frame.pack(pady=4)
        tk.Label(method_frame, text="Analysis Method:").pack(side="left", padx=5)
        tk.OptionMenu(method_frame, self.method_var, "PCA", "PaCMAP", "Both").pack(side="left")

        self.sensitivity_label = tk.Label(root, text="Sensitivity: 10")
        self.sensitivity_label.pack()
        self.sensitivity_slider = tk.Scale(root, from_=MIN_SENSITIVITY, to=MAX_SENSITIVITY,
                                           orient="horizontal", length=320, command=self.update_sensitivity_label)
        self.sensitivity_slider.set(10)
        self.sensitivity_slider.pack(pady=4)

        tk.Label(root, text="Lower = fewer segments, Higher = more segments", 
                 font=("Arial", 9), fg="gray").pack()

        # PaCMAP parameters
        pacmap_frame = tk.LabelFrame(root, text="PaCMAP Parameters", font=("Arial", 9, "bold"))
        pacmap_frame.pack(pady=6, padx=10, fill="x")
        
        neighbors_frame = tk.Frame(pacmap_frame)
        neighbors_frame.pack(pady=2)
        tk.Label(neighbors_frame, text="Neighbors:", font=("Arial", 9)).pack(side="left", padx=5)
        self.n_neighbors_var = tk.IntVar(value=10)
        self.n_neighbors_entry = tk.Entry(neighbors_frame, textvariable=self.n_neighbors_var, width=10)
        self.n_neighbors_entry.pack(side="left")
        tk.Label(neighbors_frame, text="(5-50)", font=("Arial", 8), fg="gray").pack(side="left", padx=5)
        
        ratio_frame = tk.Frame(pacmap_frame)
        ratio_frame.pack(pady=2)
        tk.Label(ratio_frame, text="MN Ratio:", font=("Arial", 9)).pack(side="left", padx=5)
        self.mn_ratio_var = tk.DoubleVar(value=0.3)
        self.mn_ratio_entry = tk.Entry(ratio_frame, textvariable=self.mn_ratio_var, width=10)
        self.mn_ratio_entry.pack(side="left")
        tk.Label(ratio_frame, text="FP Ratio:", font=("Arial", 9)).pack(side="left", padx=5)
        self.fp_ratio_var = tk.DoubleVar(value=2.0)
        self.fp_ratio_entry = tk.Entry(ratio_frame, textvariable=self.fp_ratio_var, width=10)
        self.fp_ratio_entry.pack(side="left")
        
        tk.Label(pacmap_frame, text="Iterations: 450 (fixed)", font=("Arial", 9), fg="gray").pack(pady=2)

        tk.Button(root, text="Analyze & Reorganize by Timbre", width=45, command=self.process_audio).pack(pady=6)
        tk.Button(root, text="Show Segmentation Graph", width=45, command=self.show_segmentation_graph).pack(pady=4)
        tk.Button(root, text="Show 3D Timbre Space", width=45, command=self.show_3d_timbre_space).pack(pady=4)
        tk.Button(root, text="Show 2D Timbre Comparison", width=45, command=self.show_2d_comparison).pack(pady=4)
        
        # Playback buttons
        play_frame = tk.Frame(root)
        play_frame.pack(pady=4)
        tk.Button(play_frame, text="Play PCA Reorganized", width=22, command=self.play_pca).pack(side="left", padx=2)
        tk.Button(play_frame, text="Play PaCMAP Reorganized", width=22, command=self.play_pacmap).pack(side="left", padx=2)
        
        tk.Button(root, text="STOP Audio", width=45, fg="red", command=self.stop_audio).pack(pady=4)
        
        export_frame = tk.Frame(root)
        export_frame.pack(pady=4)
        tk.Button(export_frame, text="Export PCA Audio", width=22, command=lambda: self.export_audio("PCA")).pack(side="left", padx=2)
        tk.Button(export_frame, text="Export PaCMAP Audio", width=22, command=lambda: self.export_audio("PaCMAP")).pack(side="left", padx=2)

        self.progress = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
        self.progress.pack(pady=6)

        self.segment_info = tk.Label(root, text="", font=("Arial", 9))
        self.segment_info.pack(pady=2)
        
        # Status label for buffer info
        self.buffer_info = tk.Label(root, text="Audio buffers: Not loaded", font=("Arial", 9), fg="blue")
        self.buffer_info.pack(pady=2)

    def update_sensitivity_label(self, value):
        self.sensitivity_label.config(text=f"Sensitivity: {value}")

    # --------------------- Audio Loading ---------------------
    def load_audio(self):
        path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.flac *.aiff *.aif")])
        if not path:
            return
        self.audio, self.sr = sf.read(path, always_2d=True, dtype="float32")
        self.sorted_audio_pca = None
        self.sorted_audio_pacmap = None
        self.pca_scores = None
        self.pacmap_scores = None
        
        # Clear buffers when loading new audio
        self.pca_audio_buffers = []
        self.pacmap_audio_buffers = []
        self.buffer_info.config(text="Audio buffers: Not loaded")
        
        messagebox.showinfo("Loaded", f"Loaded: {os.path.basename(path)}\nSample Rate: {self.sr}\nChannels: {self.audio.shape[1]}")

    # --------------------- Onset / Novelty / Segmentation ---------------------
    def detect_onsets(self, mono, sensitivity):
        mono = mono.astype(np.float64)
        onset_env_energy = librosa.onset.onset_strength(y=mono, sr=self.sr, hop_length=256, aggregate=np.median)
        S = np.abs(librosa.stft(mono, n_fft=2048, hop_length=256))
        spectral_flux = np.sum(np.diff(S, axis=1)**2, axis=0)
        spectral_flux = np.concatenate(([0], spectral_flux))
        spectral_flux = librosa.util.normalize(spectral_flux)
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=2048)
        hf_mask = freqs > 2000
        hf_content = np.sum(S[hf_mask, :], axis=0)
        hf_content = librosa.util.normalize(hf_content)
        hf_flux = np.concatenate(([0], np.diff(hf_content)))
        hf_flux = np.maximum(hf_flux, 0)
        hf_flux = librosa.util.normalize(hf_flux)
        combined_onset = (onset_env_energy * 0.5 + spectral_flux * 0.3 + hf_flux * 0.2)
        if np.max(combined_onset) > 0:
            combined_onset = combined_onset / np.max(combined_onset)
        base_percentile = 85 - (sensitivity ** 1.5) * 2
        base_percentile = np.clip(base_percentile, 60, 95)
        threshold = np.percentile(combined_onset, base_percentile)
        min_distance_frames = max(20, int(40 - sensitivity))
        peaks, _ = find_peaks(combined_onset, height=threshold, distance=min_distance_frames, prominence=threshold*0.2)
        backtracked_peaks = []
        window = 10
        for peak in peaks:
            start_idx = max(0, peak - window)
            segment = combined_onset[start_idx:peak+1]
            if len(segment) > 1:
                diff = np.diff(segment)
                if len(diff) > 0 and np.max(diff) > 0:
                    threshold_diff = np.max(diff) * 0.3
                    rising_points = np.where(diff > threshold_diff)[0]
                    if len(rising_points) > 0:
                        backtrack_amount = len(segment) - rising_points[0] - 1
                        backtracked_peak = peak - backtrack_amount
                        backtracked_peaks.append(backtracked_peak)
                    else:
                        backtracked_peaks.append(peak)
                else:
                    backtracked_peaks.append(peak)
            else:
                backtracked_peaks.append(peak)
        onset_frames = np.asarray(backtracked_peaks, dtype=np.int32)
        onset_samples = librosa.frames_to_samples(onset_frames, hop_length=256)
        return onset_samples

    def detect_novelty(self, mono, sensitivity):
        chroma = librosa.feature.chroma_cqt(y=mono, sr=self.sr, hop_length=512)
        R = librosa.segment.recurrence_matrix(chroma, mode='affinity', metric='cosine', width=3)
        novelty = np.sqrt(np.sum(np.diff(R, axis=1)**2, axis=0))
        novelty = np.concatenate(([0], novelty))
        novelty = librosa.util.normalize(novelty)
        percentile = 95 - (sensitivity ** 1.4) * 3
        percentile = np.clip(percentile, 70, 98)
        threshold = np.percentile(novelty, percentile)
        peaks, _ = find_peaks(novelty, height=threshold, distance=10)
        novelty_samples = librosa.frames_to_samples(peaks, hop_length=512)
        return novelty_samples

    def segment_audio_adaptive(self, mono, sensitivity, method="Onset Detection"):
        if method == "Onset Detection":
            boundaries = self.detect_onsets(mono, sensitivity)
        elif method == "Novelty Detection":
            boundaries = self.detect_novelty(mono, sensitivity)
        else:
            onset_boundaries = self.detect_onsets(mono, sensitivity)
            novelty_boundaries = self.detect_novelty(mono, sensitivity)
            boundaries = np.unique(np.concatenate([onset_boundaries, novelty_boundaries]))
        boundaries = np.concatenate(([0], boundaries, [len(mono)]))
        boundaries = np.unique(boundaries)
        min_segment_samples = int(0.25 * self.sr)
        filtered_boundaries = [boundaries[0]]
        for i in range(1, len(boundaries)):
            if boundaries[i] - filtered_boundaries[-1] >= min_segment_samples:
                filtered_boundaries.append(boundaries[i])
        if filtered_boundaries[-1] != boundaries[-1]:
            filtered_boundaries.append(boundaries[-1])
        return np.array(filtered_boundaries)

    # --------------------- MFCC Features ---------------------
    def extract_mfcc_features(self, chunk_mono):
        mfccs = librosa.feature.mfcc(y=chunk_mono, sr=self.sr, n_mfcc=14, n_fft=2048, hop_length=512)
        mfccs = mfccs[1:14,:]
        mfcc_mean = np.mean(mfccs, axis=1)
        return mfcc_mean

    def compute_pca_distance(self, mfcc_features_list):
        X = np.vstack(mfcc_features_list)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=min(3, X_scaled.shape[1]))
        X_pca = pca.fit_transform(X_scaled)
        distances = np.sqrt(np.sum(X_pca**2, axis=1))
        return distances, X_pca, pca

    def compute_pacmap_distance(self, mfcc_features_list, n_neighbors=10, MN_ratio=0.3, FP_ratio=2.0):
        if not PACMAP_AVAILABLE:
            messagebox.showerror("PaCMAP Not Available", "Install with: pip install pacmap")
            return None, None
        X = np.vstack(mfcc_features_list)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        try:
            embedding = pacmap.PaCMAP(n_components=min(3, X_scaled.shape[1]), n_neighbors=n_neighbors, MN_ratio=MN_ratio, FP_ratio=FP_ratio, num_iters=450, random_state=42, verbose=True)
            X_pacmap = embedding.fit_transform(X_scaled, init="pca")
            centroid = np.mean(X_pacmap, axis=0)
            distances = np.sqrt(np.sum((X_pacmap - centroid)**2, axis=1))
            return distances, X_pacmap
        except Exception as e:
            messagebox.showerror("PaCMAP Error", f"PaCMAP failed: {str(e)}")
            return None, None

    # --------------------- Audio Processing ---------------------
    def process_audio(self):
        if self.audio is None:
            messagebox.showwarning("No Audio", "Upload an audio file first.")
            return
        
        sensitivity = self.sensitivity_slider.get()
        method = self.segmentation_var.get()
        analysis_method = self.method_var.get()
        mono = np.mean(self.audio, axis=1).astype(np.float64)
        
        # Clear previous buffers
        self.pca_audio_buffers = []
        self.pacmap_audio_buffers = []
        self.buffer_info.config(text="Processing audio segments...")
        self.root.update_idletasks()
        
        boundaries = self.segment_audio_adaptive(mono, sensitivity, method)
        self.last_boundaries = boundaries
        chunks = []
        mfcc_features_list = []
        
        # Pre-load all audio segments into buffers
        for i in range(len(boundaries)-1):
            start, end = boundaries[i], boundaries[i+1]
            chunk = self.audio[start:end]
            chunk_mono = mono[start:end]
            if len(chunk) == 0: 
                continue
            mfcc_features_list.append(self.extract_mfcc_features(chunk_mono))
            chunks.append(chunk)
            self.progress["value"] = (i+1) * 50 / (len(boundaries)-1)  # First half of progress
            self.root.update_idletasks()
        
        # Store all chunks for instant playback
        self.all_chunks = chunks.copy()
        
        # PCA
        if analysis_method in ["PCA", "Both"]:
            distances_pca, X_pca, pca = self.compute_pca_distance(mfcc_features_list)
            self.pca_scores = X_pca
            self.pca_chunks = chunks.copy()
            
            # Create PCA audio buffers - copy data for instant playback
            self.pca_audio_buffers = [chunk.copy() for chunk in chunks]
            
            chunks_sorted = sorted(zip(distances_pca, chunks), key=lambda x: x[0])
            if self.order_var.get() == "Different → Similar":
                chunks_sorted = chunks_sorted[::-1]
            self.sorted_audio_pca = np.concatenate([c for _,c in chunks_sorted], axis=0)
            
            self.progress["value"] = 75
            self.root.update_idletasks()
        
        # PaCMAP
        if analysis_method in ["PaCMAP","Both"]:
            try:
                n_neighbors = int(self.n_neighbors_var.get())
                mn_ratio = float(self.mn_ratio_var.get())
                fp_ratio = float(self.fp_ratio_var.get())
            except:
                n_neighbors, mn_ratio, fp_ratio = 10,0.3,2.0
            
            distances_pacmap, X_pacmap = self.compute_pacmap_distance(mfcc_features_list, n_neighbors, mn_ratio, fp_ratio)
            if distances_pacmap is not None:
                self.pacmap_scores = X_pacmap
                self.pacmap_chunks = chunks.copy()
                
                # Create PaCMAP audio buffers - copy data for instant playback
                self.pacmap_audio_buffers = [chunk.copy() for chunk in chunks]
                
                chunks_sorted = sorted(zip(distances_pacmap, chunks), key=lambda x:x[0])
                if self.order_var.get() == "Different → Similar":
                    chunks_sorted = chunks_sorted[::-1]
                self.sorted_audio_pacmap = np.concatenate([c for _,c in chunks_sorted], axis=0)
            
            self.progress["value"] = 100
            self.root.update_idletasks()
        
        # Update buffer info
        total_buffers = len(self.pca_audio_buffers) + len(self.pacmap_audio_buffers)
        buffer_text = f"Audio buffers: {total_buffers} segments preloaded"
        if self.pca_audio_buffers:
            buffer_text += f" (PCA: {len(self.pca_audio_buffers)})"
        if self.pacmap_audio_buffers:
            buffer_text += f" (PaCMAP: {len(self.pacmap_audio_buffers)})"
        self.buffer_info.config(text=buffer_text)
        
        self.progress["value"] = 0

    # --------------------- Plotting / Interaction ---------------------
    def show_segmentation_graph(self):
        if self.audio is None or self.last_boundaries is None:
            messagebox.showwarning("No Segmentation", "Run analysis first.")
            return
        try:
            import matplotlib.pyplot as plt
            mono = np.mean(self.audio, axis=1)
            time = np.arange(len(mono))/self.sr
            plt.figure(figsize=(14,4))
            plt.plot(time, mono, color='steelblue')
            for b in self.last_boundaries[1:-1]:
                plt.axvline(x=b/self.sr, color='red', linestyle='--')
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.title("Segmentation")
            plt.show()
        except ImportError:
            messagebox.showerror("matplotlib required", "Install matplotlib")

    def show_3d_timbre_space(self):
        if self.pca_scores is None and self.pacmap_scores is None:
            messagebox.showwarning("No Analysis","Run analysis first")
            return
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(12,6))
            if self.pca_scores is not None:
                ax = fig.add_subplot(121, projection='3d')
                ax.scatter(self.pca_scores[:,0], self.pca_scores[:,1], self.pca_scores[:,2], c=np.linalg.norm(self.pca_scores,axis=1))
                ax.set_title("PCA Timbre Space")
            if self.pacmap_scores is not None:
                ax = fig.add_subplot(122, projection='3d')
                ax.scatter(self.pacmap_scores[:,0], self.pacmap_scores[:,1], self.pacmap_scores[:,2], c=np.linalg.norm(self.pacmap_scores - np.mean(self.pacmap_scores,axis=0),axis=1))
                ax.set_title("PaCMAP Timbre Space")
            plt.show()
        except ImportError:
            messagebox.showerror("matplotlib required","Install matplotlib")

    def highlight_selected_point(self, ax, index, method):
        """Highlight a selected point and reset previous selection"""
        # Reset previous selection
        if self.selected_index is not None and self.selected_method is not None:
            if self.selected_method in self.current_scatter_plots:
                # Clear the overlay scatter for previous selection
                for overlay in self.current_scatter_plots[self.selected_method].get("overlays", []):
                    overlay.remove()
                if "overlays" in self.current_scatter_plots[self.selected_method]:
                    self.current_scatter_plots[self.selected_method]["overlays"] = []
        
        # Highlight new selection
        scores = self.pca_scores if method == "PCA" else self.pacmap_scores
        if scores is not None and 0 <= index < len(scores):
            self.selected_index = index
            self.selected_method = method
            
            # Create a highlighted point (bigger, red border)
            x, y = scores[index, 0], scores[index, 1]
            highlight = ax.scatter([x], [y], s=200, c='red', edgecolors='black', 
                                  linewidths=3, alpha=0.8, zorder=5)
            
            # Create a pulsing effect with a semi-transparent circle
            pulse = ax.scatter([x], [y], s=300, facecolors='none', edgecolors='red', 
                              linewidths=2, alpha=0.5, zorder=4)
            
            # Store overlays for later removal
            if method not in self.current_scatter_plots:
                self.current_scatter_plots[method] = {}
            if "overlays" not in self.current_scatter_plots[method]:
                self.current_scatter_plots[method]["overlays"] = []
            
            self.current_scatter_plots[method]["overlays"].extend([highlight, pulse])
            
            # Add text annotation
            text = ax.text(x, y + 0.02, f"Segment {index+1}", fontsize=9, 
                          ha='center', va='bottom', color='red', weight='bold')
            self.current_scatter_plots[method]["overlays"].append(text)
            
            # Redraw
            ax.figure.canvas.draw_idle()

    def show_2d_comparison(self):
        if self.pca_scores is None and self.pacmap_scores is None:
            messagebox.showwarning("No Analysis","Run analysis first")
            return
        
        # Check if audio buffers are loaded
        if not self.pca_audio_buffers and not self.pacmap_audio_buffers:
            messagebox.showwarning("No Audio Buffers", "Run analysis first to preload audio segments")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            # Clear previous selections
            self.selected_index = None
            self.selected_method = None
            self.current_scatter_plots = {}
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.canvas.manager.set_window_title('2D Timbre Space - Click on points to play audio (Instant playback)')
            
            # Plot PCA
            if self.pca_scores is not None and self.pca_audio_buffers:
                norm_pca = np.linalg.norm(self.pca_scores, axis=1)
                scatter_pca = axes[0].scatter(
                    self.pca_scores[:, 0], 
                    self.pca_scores[:, 1], 
                    c=norm_pca, 
                    cmap='viridis',
                    s=50,  # Base size
                    alpha=0.7,
                    picker=True,  # Enable picking
                    pickradius=10  # Increase pick radius for easier clicking
                )
                axes[0].set_title("PCA 2D Timbre Space (Click points to play)")
                axes[0].set_xlabel("Component 1")
                axes[0].set_ylabel("Component 2")
                axes[0].grid(True, alpha=0.3)
                
                # Store reference to the scatter plot
                self.current_scatter_plots["PCA"] = {
                    "scatter": scatter_pca, 
                    "ax": axes[0],
                    "buffers": self.pca_audio_buffers
                }
            
            # Plot PaCMAP
            if self.pacmap_scores is not None and self.pacmap_audio_buffers:
                centroid = np.mean(self.pacmap_scores, axis=0)
                norm_pacmap = np.linalg.norm(self.pacmap_scores - centroid, axis=1)
                scatter_pacmap = axes[1].scatter(
                    self.pacmap_scores[:, 0], 
                    self.pacmap_scores[:, 1], 
                    c=norm_pacmap, 
                    cmap='plasma',
                    s=50,  # Base size
                    alpha=0.7,
                    picker=True,  # Enable picking
                    pickradius=10  # Increase pick radius for easier clicking
                )
                axes[1].set_title("PaCMAP 2D Timbre Space (Click points to play)")
                axes[1].set_xlabel("Component 1")
                axes[1].set_ylabel("Component 2")
                axes[1].grid(True, alpha=0.3)
                
                # Store reference to the scatter plot
                self.current_scatter_plots["PaCMAP"] = {
                    "scatter": scatter_pacmap, 
                    "ax": axes[1],
                    "buffers": self.pacmap_audio_buffers
                }
            
            # Add colorbars
            if self.pca_scores is not None and self.pca_audio_buffers:
                plt.colorbar(scatter_pca, ax=axes[0], label='Distance from origin')
            if self.pacmap_scores is not None and self.pacmap_audio_buffers:
                plt.colorbar(scatter_pacmap, ax=axes[1], label='Distance from centroid')
            
            plt.tight_layout()
            
            def onpick(event):
                """Handle point selection - INSTANT PLAYBACK from buffers"""
                start_time = time.time()
                ind = event.ind[0]
                
                # Determine which method was clicked
                if event.artist == self.current_scatter_plots.get("PCA", {}).get("scatter"):
                    method = "PCA"
                    ax_index = 0
                elif event.artist == self.current_scatter_plots.get("PaCMAP", {}).get("scatter"):
                    method = "PaCMAP"
                    ax_index = 1
                else:
                    return
                
                # Get the audio buffer
                buffers = self.current_scatter_plots[method]["buffers"]
                if ind < len(buffers):
                    segment = buffers[ind]  # INSTANT access from preloaded buffer
                    
                    # Highlight the selected point
                    self.highlight_selected_point(axes[ax_index], ind, method)
                    
                    # Update segment info in GUI
                    segment_duration = len(segment) / self.sr
                    self.segment_info.config(
                        text=f"Playing: {method} Segment {ind+1}/{len(buffers)} ({segment_duration:.2f}s)"
                    )
                    
                    # Play the audio segment INSTANTLY
                    if self.current_playback_thread and self.current_playback_thread.is_alive():
                        sd.stop()  # Stop any currently playing audio
                    
                    self.current_playback_thread = threading.Thread(
                        target=lambda: sd.play(segment, self.sr, blocking=False),
                        daemon=True
                    )
                    self.current_playback_thread.start()
                    
                    # Debug timing
                    click_time = (time.time() - start_time) * 1000
                    print(f"Click to playback: {click_time:.1f}ms")
                
            def on_key(event):
                """Handle keyboard events"""
                if event.key == 'escape':
                    # Clear selection on escape
                    if self.selected_index is not None and self.selected_method is not None:
                        if self.selected_method in self.current_scatter_plots:
                            ax = self.current_scatter_plots[self.selected_method]["ax"]
                            for overlay in self.current_scatter_plots[self.selected_method].get("overlays", []):
                                overlay.remove()
                            self.current_scatter_plots[self.selected_method]["overlays"] = []
                            ax.figure.canvas.draw_idle()
                            self.selected_index = None
                            self.selected_method = None
                            self.segment_info.config(text="")
                elif event.key == ' ':
                    # Space to stop audio
                    self.stop_audio()
            
            # Connect event handlers
            fig.canvas.mpl_connect('pick_event', onpick)
            fig.canvas.mpl_connect('key_press_event', on_key)
            
            # Add instructions to the plot
            plt.figtext(0.5, 0.01, 
                       "Click on points to play audio segments (instant) | Press 'ESC' to clear selection | SPACE to stop audio", 
                       ha='center', fontsize=9, style='italic')
            
            plt.show()
            
        except ImportError:
            messagebox.showerror("matplotlib required","Install matplotlib")

    # --------------------- Playback ---------------------
    def play_audio(self, audio_data):
        if audio_data is None:
            messagebox.showwarning("No Audio","No audio to play")
            return
        
        # Stop any currently playing audio
        if self.current_playback_thread and self.current_playback_thread.is_alive():
            sd.stop()
        
        self.current_playback_thread = threading.Thread(
            target=lambda: sd.play(audio_data, self.sr, blocking=False),
            daemon=True
        )
        self.current_playback_thread.start()

    def play_original(self): 
        self.play_audio(self.audio)
    
    def play_pca(self): 
        self.play_audio(self.sorted_audio_pca)
    
    def play_pacmap(self): 
        self.play_audio(self.sorted_audio_pacmap)
    
    def stop_audio(self): 
        sd.stop()
        self.segment_info.config(text="Audio stopped")

    # --------------------- Export ---------------------
    def export_audio(self, method="PCA"):
        if method=="PCA" and self.sorted_audio_pca is not None:
            filename = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV","*.wav")])
            if filename: sf.write(filename, self.sorted_audio_pca, self.sr)
        elif method=="PaCMAP" and self.sorted_audio_pacmap is not None:
            filename = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV","*.wav")])
            if filename: sf.write(filename, self.sorted_audio_pacmap, self.sr)
        else:
            messagebox.showwarning("No Audio","Run analysis first")

if __name__=="__main__":
    root = tk.Tk()
    app = TimbreReorganizerPCA(root)
    root.mainloop()