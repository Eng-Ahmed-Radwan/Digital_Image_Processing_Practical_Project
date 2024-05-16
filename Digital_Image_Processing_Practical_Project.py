import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageFilter, ImageEnhance
import cv2
import numpy as np

class ImageEditorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Editor")

        # Image variables
        self.original_image = None
        self.filtered_image = None
        self.current_image = None

        # Filter variable
        self.selected_filter = tk.StringVar(self)
        self.selected_filter.set("Original Image")  # default value

        # Brightness and Contrast variables
        self.brightness_value = tk.DoubleVar(self)
        self.brightness_value.set(1.0)  # default value
        self.contrast_value = tk.DoubleVar(self)
        self.contrast_value.set(1.0)  # default value

        # Border width
        self.border_width = 5

        # Create widgets
        self.create_widgets()

    def create_widgets(self):
        # Style
        self.style = ttk.Style()
        self.style.theme_use("clam")

        # Menu
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open", command=self.load_image)
        file_menu.add_command(label="Save", command=self.save_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)

        # Frame
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Upload Button
        self.upload_button = ttk.Button(main_frame, text="Upload Image", command=self.load_image)
        self.upload_button.grid(row=0, column=0, padx=10, pady=10)

        # Filter Option Menu
        self.filter_options = [
            "Original Image", "HPF", "LPF", "Mean", "Median", "Roberts", "Prewitt", "Sobel",
            "Threshold Segmentation", "Hough Circle Transform",
            "Erosion", "Dilation", "Opening", "Closing"
        ]
        filter_label = ttk.Label(main_frame, text="Select Filter:")
        filter_label.grid(row=0, column=1, padx=10, pady=10)
        self.filter_option_menu = ttk.OptionMenu(main_frame, self.selected_filter, *self.filter_options, command=self.apply_filter)
        self.filter_option_menu.grid(row=0, column=2, padx=10, pady=10)

        # Brightness Label and Scale
        self.brightness_label = ttk.Label(main_frame, text="Brightness:")
        self.brightness_label.grid(row=1, column=0, padx=10, pady=5)
        self.brightness_scale = tk.Scale(main_frame, from_=0.5, to=1.5, resolution=0.05, orient=tk.HORIZONTAL, variable=self.brightness_value, command=self.apply_filter)
        self.brightness_scale.set(1.0)  # Set default brightness value
        self.brightness_scale.grid(row=1, column=1, columnspan=2, padx=10, pady=5)

        # Contrast Label and Scale
        self.contrast_label = ttk.Label(main_frame, text="Contrast:")
        self.contrast_label.grid(row=2, column=0, padx=10, pady=5)
        self.contrast_scale = tk.Scale(main_frame, from_=0.5, to=1.5, resolution=0.05, orient=tk.HORIZONTAL, variable=self.contrast_value, command=self.apply_filter)
        self.contrast_scale.set(1.0)  # Set default contrast value
        self.contrast_scale.grid(row=2, column=1, columnspan=2, padx=10, pady=5)

        # Canvas for displaying images
        self.canvas_before = tk.Canvas(main_frame, bg="white", bd=2, relief=tk.SOLID)
        self.canvas_before.grid(row=3, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")

        self.canvas_after = tk.Canvas(main_frame, bg="white", bd=2, relief=tk.SOLID)
        self.canvas_after.grid(row=3, column=3, columnspan=3, padx=10, pady=10, sticky="nsew")

        # Save Button to save the edited image
        self.save_button = ttk.Button(main_frame, text="Save Image", command=self.save_image, state=tk.DISABLED)
        self.save_button.grid(row=4, column=0, columnspan=6, padx=10, pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.gif")])
        if file_path:
            try:
                self.original_image = Image.open(file_path)
                self.display_image_before(self.original_image)
                self.save_button.config(state=tk.NORMAL)  # Enable save button after successful load
                self.reset_filter()  # Reset filter upon loading a new image
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")

    def display_image_before(self, image):
        self.current_image = image
        width, height = image.size
        aspect_ratio = width / height
        new_width = min(width, self.winfo_width() // 3)
        new_height = int(new_width / aspect_ratio)
        resized_image = image.resize((new_width, new_height), resample=Image.BILINEAR)
        self.photo_before = ImageTk.PhotoImage(resized_image)
        self.canvas_before.delete("all")
        self.canvas_before.create_image(0, 0, anchor="nw", image=self.photo_before)
        self.canvas_before.config(width=new_width + 2 * self.border_width, height=new_height + 2 * self.border_width)

    def display_image_after(self, image):
        width, height = image.size
        aspect_ratio = width / height
        new_width = min(width, self.winfo_width() // 3)
        new_height = int(new_width / aspect_ratio)
        resized_image = image.resize((new_width, new_height), resample=Image.BILINEAR)
        self.photo_after = ImageTk.PhotoImage(resized_image)
        self.canvas_after.delete("all")
        self.canvas_after.create_image(0, 0, anchor="nw", image=self.photo_after)
        self.canvas_after.config(width=new_width + 2 * self.border_width, height=new_height + 2 * self.border_width)

    def apply_filter(self, *args):
        if self.original_image is None:
            messagebox.showerror("Error", "No image loaded")
            return

        selected_filter = self.selected_filter.get()
        if selected_filter == "Original Image":
            self.filtered_image = self.original_image.copy()  # Make a copy of the original image
        elif selected_filter == "HPF":
            self.apply_hpf()
        elif selected_filter == "LPF":
            self.apply_lpf()
        elif selected_filter == "Mean":
            self.apply_mean_filter()
        elif selected_filter == "Median":
            self.apply_median_filter()
        elif selected_filter == "Roberts":
            self.apply_roberts_edge_detector()
        elif selected_filter == "Prewitt":
            self.apply_prewitt_edge_detector()
        elif selected_filter == "Sobel":
            self.apply_sobel_edge_detector()
        elif selected_filter == "Threshold Segmentation":
            self.apply_thresholding_segmentation()
        elif selected_filter == "Hough Circle Transform":
            self.apply_hough_circle_transform()
        elif selected_filter == "Erosion":
            self.apply_erosion()
        elif selected_filter == "Dilation":
            self.apply_dilation()
        elif selected_filter == "Opening":
            self.apply_open()
        elif selected_filter == "Closing":
            self.apply_close()

        self.display_image_after(self.filtered_image)

    def reset_filter(self):
        self.selected_filter.set("Original Image")  # Reset filter option to "Original Image"
        self.brightness_scale.set(1.0)  # Reset brightness scale to default value
        self.contrast_scale.set(1.0)  # Reset contrast scale to default value
        self.apply_filter()  # Apply reset filter

    def save_image(self):
        if self.filtered_image:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
            if file_path:
                self.filtered_image.save(file_path)
                messagebox.showinfo("Success", "Image saved successfully.")
        else:
            messagebox.showerror("Error", "No filtered image to save.")

    def apply_hpf(self):
        # Apply high-pass filter (HPF) to enhance edges
        gray_image = self.original_image.convert("L")  # Convert to grayscale
        hpf_image = gray_image.filter(ImageFilter.FIND_EDGES)
        self.filtered_image = hpf_image

    def apply_lpf(self):
        # Apply low-pass filter (LPF) for blurring
        lpf_image = self.original_image.filter(ImageFilter.BLUR)
        self.filtered_image = lpf_image

    def apply_mean_filter(self):
        # Apply mean filter to smooth the image
        mean_image = self.original_image.filter(ImageFilter.SMOOTH)
        self.filtered_image = mean_image

    def apply_median_filter(self):
        # Apply median filter to remove noise
        median_image = self.original_image.filter(ImageFilter.MedianFilter)
        self.filtered_image = median_image

    def apply_roberts_edge_detector(self):
        # Apply Roberts edge detector to detect edges in the image
        gray_image = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_BGR2GRAY)
        roberts_image = cv2.Canny(gray_image, 100, 200)
        self.filtered_image = Image.fromarray(roberts_image)

    def apply_prewitt_edge_detector(self):
        # Apply Prewitt edge detector to detect edges in the image
        gray_image = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_BGR2GRAY)
        prewitt_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        prewitt_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        prewitt_image = np.sqrt(prewitt_x ** 2 + prewitt_y ** 2)
        prewitt_image = cv2.normalize(prewitt_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.filtered_image = Image.fromarray(prewitt_image)

    def apply_sobel_edge_detector(self):
        # Apply Sobel edge detector to detect edges in the image
        gray_image = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_image = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        sobel_image = cv2.normalize(sobel_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.filtered_image = Image.fromarray(sobel_image)

    def apply_thresholding_segmentation(self):
        threshold_value = 127  # Example threshold value
        gray_image = self.original_image.convert("L")  # Convert to grayscale
        threshold_image = gray_image.point(lambda p: p > threshold_value and 255)
        self.filtered_image = threshold_image

    def apply_hough_circle_transform(self):
        gray_image = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            hough_image = np.array(self.original_image).copy()
            for i in circles[0, :]:
                cv2.circle(hough_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(hough_image, (i[0], i[1]), 2, (0, 0, 255), 3)
            self.filtered_image = Image.fromarray(hough_image)

    def apply_erosion(self):
        kernel = np.ones((3, 3), np.uint8)
        erosion_image = cv2.erode(np.array(self.original_image), kernel, iterations=1)
        self.filtered_image = Image.fromarray(erosion_image)

    def apply_dilation(self):
        kernel = np.ones((3, 3), np.uint8)
        dilation_image = cv2.dilate(np.array(self.original_image), kernel, iterations=1)
        self.filtered_image = Image.fromarray(dilation_image)

    def apply_open(self):
        kernel = np.ones((3, 3), np.uint8)
        open_image = cv2.morphologyEx(np.array(self.original_image), cv2.MORPH_OPEN, kernel)
        self.filtered_image = Image.fromarray(open_image)

    def apply_close(self):
        kernel = np.ones((3, 3), np.uint8)
        close_image = cv2.morphologyEx(np.array(self.original_image), cv2.MORPH_CLOSE, kernel)
        self.filtered_image = Image.fromarray(close_image)

if __name__ == "__main__":
    app = ImageEditorApp()
    app.mainloop()
