print("Loading Libraries")
print("Please Wait...")

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import queue
import os
import torch
from diffusers import Flux2KleinPipeline
from diffusers.utils import load_image


class FluxImageGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Flux2 Local Image Generator")
        self.root.geometry("835x825")
        self.is_generating = False
        self.result_queue = queue.Queue()
        
        # State variables
        self.model_path = None
        self.input_image_pil = None
        self.output_image_path = "generated_image.png"
        self.pipeline = None
        self.is_model_loaded = False
        self.is_loading = False
        self.cancel_requested = False
        self.output_counter = 0  # Track generated images
        self.output_folder = "Output"
        self.output_image_path = os.path.join(self.output_folder, "generated_image.png")       
        
        self.setup_ui()
        self.root.after(100, self.check_queue)

        # Auto-detect model folder in same directory as script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_folder = os.path.join(script_dir, "FLUX.2-klein-4B")
        if os.path.exists(model_folder):
            self.model_path = model_folder
            self.model_entry.delete(0, tk.END)
            self.model_entry.insert(0, model_folder)
            print(f"✅ Auto-detected model folder: {model_folder}")
            # Trigger model loading (same logic as select_model_folder)
            if not self.is_loading and not self.is_model_loaded:
                self.is_loading = True
                self.status_label.config(text="Status: Loading model...", foreground="orange")
                threading.Thread(target=self._load_model_in_background, args=(model_folder,), daemon=True).start()
        else:
            print("⚠️ Model folder 'FLUX.2-klein-4B' not found in script directory.")

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # === Model Selection ===
        ttk.Label(main_frame, text="Model Folder:").grid(row=0, column=0, sticky="w", pady=(0, 5))
        self.model_entry = ttk.Entry(main_frame, width=50)
        self.model_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=(0, 5))
        ttk.Button(main_frame, text="📁 Browse", command=self.select_model_folder).grid(row=0, column=2, sticky="w", pady=(0, 5))

        # === Input Image ===
        ttk.Label(main_frame, text="Input Image (Optional):").grid(row=1, column=0, sticky="w", pady=(8, 2))
        self.input_image_entry = ttk.Entry(main_frame, width=45)
        self.input_image_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=(5, 2))
        self.input_img_button = ttk.Button(main_frame, text="📁 Browse", command=self.toggle_input_image)
        self.input_img_button.grid(row=1, column=2, sticky="w", pady=(5, 2))

        # === Resolution & Auto-Detect ===
        ttk.Label(main_frame, text="Resolution:").grid(row=2, column=0, sticky="w", pady=(10, 2))
        self.width_entry = ttk.Entry(main_frame, width=5, justify="right")
        self.width_entry.insert(0, "512")
        self.width_entry.grid(row=2, column=1, sticky="w", padx=(5, 5), pady=(10, 2))
        ttk.Label(main_frame, text="x").grid(row=2, column=1, sticky="w", padx=60, pady=(10, 2))
        self.height_entry = ttk.Entry(main_frame, width=5, justify="right")
        self.height_entry.insert(0, "512")
        self.height_entry.grid(row=2, column=1, sticky="w", padx=(80, 0), pady=(10, 2))
        
        self.auto_res_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(main_frame, text="Auto-detect from input image", variable=self.auto_res_var, command=self.update_auto_res).grid(row=2, column=1, sticky="w", padx=(140, 0), pady=(10, 2))

        # === Inference Steps ===
        ttk.Label(main_frame, text="Inference Steps:").grid(row=3, column=0, sticky="w", pady=(10, 2))
        self.steps_var = tk.IntVar(value=8)
        ttk.Radiobutton(main_frame, text="4 (Low)", variable=self.steps_var, value=4).grid(row=3, column=1, sticky="w", padx=(10, 0), pady=(10, 2))
        ttk.Radiobutton(main_frame, text="8 (Medium)", variable=self.steps_var, value=8).grid(row=3, column=1, sticky="w", padx=(100, 0), pady=(10, 2))
        ttk.Radiobutton(main_frame, text="16 (High)", variable=self.steps_var, value=16).grid(row=3, column=1, sticky="w", padx=(210, 0), pady=(10, 2))

        # === Seed ===
        ttk.Label(main_frame, text="Seed Number (blank=Random):").grid(row=4, column=0, sticky="w", pady=(10, 2))
        self.seed_entry = ttk.Entry(main_frame, width=11, justify="right")
        self.seed_entry.grid(row=4, column=1, sticky="w", padx=5, pady=(10, 2))
        self._setup_seed_menu()  # right click menu
        
        # === Prompt ===
        ttk.Label(main_frame, text="Prompt:").grid(row=5, column=0, sticky="nw", pady=(10, 2))
        self.prompt_text = tk.Text(main_frame, height=5, width=62, wrap="word", undo=True)
        self.prompt_text.grid(row=5, column=1, columnspan=2, sticky="w", padx=5, pady=(14, 2))
        self.prompt_text.insert("1.0", "A cyberpunk cityscape at night with neon lights, highly detailed, cinematic lighting")
        self._setup_prompt_menu()  # right click menu
        self.prompt_text.bind("<Return>", self._on_enter_key)  # Enter → Generate
        self.prompt_text.bind("<Shift-Return>", self._on_shift_enter)  # Shift+Enter → New line

        # === Generate Button ===
        self.gen_button = ttk.Button(main_frame, text="🚀 Generate", command=self.run_generation_thread)
        self.gen_button.grid(row=6, column=0, columnspan=3, pady=(10, 2))

        # === Status Label ===
        self.status_label = ttk.Label(main_frame, text="Status: Ready", foreground="gray")
        self.status_label.grid(row=7, column=0, columnspan=3, pady=(2, 2))

        # === Image Previews ===
        ttk.Separator(main_frame).grid(row=8, column=0, columnspan=3, sticky="ew", pady=8)
        
        preview_frame = ttk.Frame(main_frame)
        preview_frame.grid(row=9, column=0, columnspan=3, sticky="nsew")
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.columnconfigure(1, weight=1)
        
        ttk.Label(preview_frame, text="Input Preview", font=("Segoe UI", 9, "italic")).grid(row=0, column=0, pady=(0, 5))
        self.input_canvas = tk.Canvas(preview_frame, width=400, height=400, bg="#f0f0f0", highlightthickness=1, highlightbackground="#ccc")
        self.input_canvas.grid(row=1, column=0, padx=(0, 10))
        
        ttk.Label(preview_frame, text="Output Preview", font=("Segoe UI", 9, "italic")).grid(row=0, column=1, pady=(0, 5))
        self.output_canvas = tk.Canvas(preview_frame, width=400, height=400, bg="#f0f0f0", highlightthickness=1, highlightbackground="#ccc")
        self.output_canvas.grid(row=1, column=1)

        self.root.grid_rowconfigure(9, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Helper for thread-safe UI updates
        def _update_status(text, color):
            self.status_label.config(text=text, foreground=color)
        self._update_status = _update_status


    def select_model_folder(self):
        path = filedialog.askdirectory(title="Select Flux Model Folder")
        if path:
            self.model_path = path
            self.model_entry.delete(0, tk.END)
            self.model_entry.insert(0, os.path.normpath(path))

            if self.is_loading:
                print("⚠️ Please wait for the current model to finish loading.")
                return

            # Reset state to allow reloading
            self.is_model_loaded = False
            self.pipeline = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Free VRAM from old model
            
            self.is_loading = True
            self.status_label.config(text="Status: Loading model...", foreground="orange")
            threading.Thread(target=self._load_model_in_background, args=(path,), daemon=True).start()



    def select_input_image(self):
        path = filedialog.askopenfilename(
            title="Select Input Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.webp *.gif")]
        )
        if path:
            self.input_image_path = path
            self.input_image_entry.delete(0, tk.END)
            self.input_image_entry.insert(0, os.path.normpath(path))
            try:
                self.input_image_pil = load_image(path)
                self.update_preview(self.input_image_pil, self.input_canvas)
                print(f"✅ Input image loaded: {path}")
                # Change button text to cancel
                self.input_img_button.config(text="🗑 Cancel")
            except Exception as e:
                print(f"❌ Failed to load image: {e}")
            if self.auto_res_var.get():
                self.update_auto_res()
                
    def toggle_input_image(self):
        if self.input_image_pil is None:
            self.select_input_image()
        else:
            self.input_image_pil = None
            self.input_image_entry.delete(0, tk.END)
            self.update_preview(None, self.input_canvas)
            self.input_img_button.config(text="📁 Browse")
            print("🗑️ Input image cleared.")


    def _load_model_in_background(self, model_path):
        try:
            print(f"🎨 Loading pipeline from: {model_path}")
            print("⏳ This may take a moment...")
            
            pipeline = Flux2KleinPipeline.from_pretrained(
                model_path,
                local_files_only=True,
            )
            print("🏋  Weights loaded")
            print("🔌 Setting up device...")
            
            # Auto-detect best dtype for your GPU
            if torch.cuda.is_available():
                device = torch.device("cuda")
                # Use bfloat16 for newer NVIDIA GPUs, float16 for older
                if torch.cuda.get_device_capability()[0] >= 8:  # Volta+ (A100, V100, RTX 30/40 series)
                    dtype = torch.bfloat16
                else:
                    dtype = torch.float16
                pipeline = pipeline.to(dtype).to(device)
                print(f"🖥️ Device: CUDA | Dtype: {dtype}")
                print("💾 Model loading on GPU")
            else:
                pipeline = pipeline.to(torch.float32)
                print(f"💻 Device: CPU | Dtype: float32")
                print("💾 Model loading on CPU")
    
            pipeline.enable_model_cpu_offload()
            print("💻 Model partially offloaded to CPU for VRAM efficiency")
                
            self.pipeline = pipeline
            self.is_model_loaded = True
            self.is_loading = False
            self._update_status("Status: Model Ready", "green")
            print("✅ Model fully loaded and ready!")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.is_loading = False
            self._update_status("Status: Load Failed", "red")
            # Show error dialog too:
            messagebox.showerror("Load Failed", f"Could not load model:\n{str(e)[:100]}")


    def update_auto_res(self):
        if self.auto_res_var.get() and self.input_image_pil:
            w, h = self.input_image_pil.size
            self.width_entry.delete(0, tk.END)
            self.height_entry.delete(0, tk.END)
            self.width_entry.insert(0, str(w))
            self.height_entry.insert(0, str(h))
            print(f"📐 Auto-detected resolution: {w}x{h}")

    def update_preview(self, pil_image, canvas, max_size=(401, 401)):
        canvas.delete("all")
        if pil_image is None:
            canvas.create_text(200, 200, text="No Image", fill="#888")
            return
        
        # Work on a copy to preserve the original resolution
        preview_img = pil_image.copy()
        preview_img.thumbnail(max_size, Image.LANCZOS)
        
        img_tk = ImageTk.PhotoImage(preview_img)
        canvas.create_image(200, 200, anchor="center", image=img_tk)
        canvas.img_tk = img_tk  # Keep reference to prevent garbage collection


    def run_generation_thread(self):
        if self.is_generating:
            return
        # Check if model folder has been selected at all
        if not self.model_path:
            messagebox.showerror("Not Ready", "Model not loaded, please select a model.")
            return

        # Check if model is still loading
        if self.is_loading:
            messagebox.showerror("Not Ready", "Please wait for the model to finish loading.")
            return
            
        self.is_generating = True
        self.cancel_requested = False
        self.gen_button.config(state="normal", text="🛑 Cancel", command=self.cancel_generation)
        self._update_status("Status: Generating...", "blue")

        
        prompt = self.prompt_text.get("1.0", tk.END).strip()
        steps = self.steps_var.get()
        seed_str = self.seed_entry.get().strip()
        seed = int(seed_str) if seed_str else None
        width = int(self.width_entry.get())
        height = int(self.height_entry.get())
        input_img = self.input_image_pil
        
            
        # Create Output folder if it doesn't exist
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            print(f"📁 Created output folder: {self.output_folder}")        

        # Find highest existing image number before generating
        if os.path.exists(self.output_folder):
            existing_files = [f for f in os.listdir(self.output_folder) 
                             if f.startswith("generated_image_") and f.endswith(".png")]
            if existing_files:
                # Extract numbers from filenames
                numbers = []
                for f in existing_files:
                    # Remove "generated_image_" and ".png"
                    num_str = f.replace("generated_image_", "").replace(".png", "")
                    try:
                        numbers.append(int(num_str))
                    except ValueError:
                        continue
                if numbers:
                    # Increment from highest existing
                    self.output_counter = max(numbers) + 1
                else:
                    self.output_counter = 0
            else:
                self.output_counter = 0
        else:
            self.output_counter = 0

        # update output image with increment
        filename = f"generated_image_{self.output_counter:04d}.png"
        self.output_image_path = os.path.join(self.output_folder, filename)
        print(f"🧵 Starting generation thread. Counter: {self.output_counter}")
        # Pass the already-loaded pipeline instead of reloading
        threading.Thread(target=self._generate_worker, args=(self.pipeline, prompt, steps, seed, width, height, input_img), daemon=True).start()


    def _generate_worker(self, pipeline, prompt, steps, seed, width, height, input_img):
        try:
            seed_str = self.seed_entry.get().strip()
            seed = int(seed_str) if seed_str else None
            
            # If random, generate and display the seed
            if seed is None:
                import random
                seed = random.randint(0, 2**31 - 1)
                self.seed_entry.delete(0, tk.END)
                self.seed_entry.insert(0, str(seed))
                print(f"🎲 Random seed generated")
            
            # Set the seed
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                print(f"🎲 Seed fixed to: {seed}")


            print(f"⚙️ Generating: {width}x{height} | Steps: {steps} | Prompt: {prompt[:50]}...")
            
            # Generate (we can't interrupt mid-pipeline, but we check after)
            output = pipeline(
                prompt=prompt,
                image=input_img,
                width=width,
                height=height,
                num_inference_steps=steps
            )

            # FEIGN CANCELLATION: Check if user cancelled AFTER pipeline completes
            if self.cancel_requested:
                print("⚠️ Generation completed, but user cancelled - discarding result")
                self.result_queue.put(("done", None))
                return  # Don't save or preview the image
            
            # Save the image
            img = output.images[0]
            img.save(self.output_image_path)
            print(f"✅ Image saved to: {self.output_image_path}")
            
            self.result_queue.put(("success", img))
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            self.result_queue.put(("error", str(e)))
        finally:
            self.result_queue.put(("done", None))


    def check_queue(self):
        try:
            while True:
                msg_type, data = self.result_queue.get_nowait()
                if msg_type == "success" and self.cancel_requested:
                    # Skip preview if cancelled
                    print("⚠️ Cancelled - skipping preview update")
                    self.cancel_requested = False
                    continue
                elif msg_type == "success":
                    self.update_preview(data, self.output_canvas)
                elif msg_type == "error":
                    messagebox.showerror("Generation Failed", f"Error:\n{data}")
                elif msg_type == "done":
                    self.reset_gen_button()
        except queue.Empty:
            pass
        self.root.after(100, self.check_queue)


    def reset_gen_button(self):
        self.is_generating = False
        self.gen_button.config(state="normal", text="🚀 Generate", command=self.run_generation_thread)
        # Only reset status to green if model is actually loaded
        if self.is_model_loaded and not self.is_loading:
            self._update_status("Status: Model Ready 🚀", "green")
        elif not self.is_model_loaded:
            self._update_status("Status: Waiting for model", "gray")
            
    def cancel_generation(self):
        self.cancel_requested = True
        self.is_generating = False
        self.gen_button.config(state="normal", text="🛑 Canceling...", command=self.cancel_generation)
        self._update_status("Status: Canceling (generation will complete in background)...", "orange")
        print("🛑 User requested cancellation (discarding result on completion).")
        self.root.after(100, self.check_queue)


    # right click menu
    def _setup_prompt_menu(self):
        self.prompt_menu = tk.Menu(self.prompt_text, tearoff=False)
        self.prompt_menu.add_command(label="Cut", command=lambda: self.prompt_text.event_generate("<<Cut>>"))
        self.prompt_menu.add_command(label="Copy", command=lambda: self.prompt_text.event_generate("<<Copy>>"))
        self.prompt_menu.add_command(label="Paste", command=lambda: self.prompt_text.event_generate("<<Paste>>"))
        self.prompt_menu.add_command(label="Delete", command=lambda: self.prompt_text.event_generate("<<Clear>>"))
        self.prompt_menu.add_command(label="Select All", command=lambda: self.prompt_text.event_generate("<<SelectAll>>"))
        self.prompt_menu.add_separator()
        self.prompt_menu.add_command(label="Undo", command=lambda: self.prompt_text.event_generate("<<Undo>>"))
        self.prompt_menu.add_command(label="Redo", command=lambda: self.prompt_text.event_generate("<<Redo>>"))
        self.prompt_text.bind("<Button-3>", self._show_prompt_menu)        

    def _show_prompt_menu(self, event):
        self.prompt_menu.post(event.x_root, event.y_root)
        
    def _setup_seed_menu(self):
        self.seed_menu = tk.Menu(self.seed_entry, tearoff=False)
        self.seed_menu.add_command(label="Cut", command=lambda: self.seed_entry.event_generate("<<Cut>>"))
        self.seed_menu.add_command(label="Copy", command=lambda: self.seed_entry.event_generate("<<Copy>>"))
        self.seed_menu.add_command(label="Paste", command=lambda: self.seed_entry.event_generate("<<Paste>>"))
        self.seed_menu.add_command(label="Delete", command=lambda: self.seed_entry.event_generate("<<Clear>>"))
        self.seed_menu.add_command(label="Select All", command=lambda: self.seed_entry.event_generate("<<SelectAll>>"))
        self.seed_entry.bind("<Button-3>", self._show_seed_menu)        

    def _show_seed_menu(self, event):
        self.seed_menu.post(event.x_root, event.y_root)

    def _on_enter_key(self, event):
        """Trigger generation when Enter is pressed, or cancel if generating"""
        if self.is_generating:
            # Cancel generation
            self.cancel_generation()
        elif not self.is_loading:
            # Start generation
            self.run_generation_thread()
        return "break"

    def _on_shift_enter(self, event):
        """Insert newline when Shift+Enter is pressed"""
        self.prompt_text.insert(tk.INSERT, "\n")
        return "break"


if __name__ == "__main__":
    root = tk.Tk()
    app = FluxImageGeneratorGUI(root)
    root.mainloop()
