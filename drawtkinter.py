import tkinter as tk
import numpy as np
from PIL import ImageGrab
from math import ceil
# Function to draw rectangles
import os
from torch.nn import MaxPool2d
import torch
import argparse
def change_class(index):
    global current_class
    current_class = index

# Function to enable eraser mode
def enable_eraser():
    global current_class
    current_class = 255

def on_release(event):
    end_x, end_y = min(event.x, width), min(event.y, height)
    x1, y1, x2, y2 = start_x, start_y, end_x, end_y
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    if current_class != 255:
        if class_names[current_class] == "wall":
            if abs(end_x - start_x) > abs(end_y - start_y):
                y2 = y1  # Make the rectangle a horizontal line
            else:
                x2 = x1
            canvas.create_line(x1, y1, x2, y2, fill=colors[current_class],width=scale)
            update_array_wall(x1, y1, x2, y2, current_class)
        else:
            canvas.create_rectangle(x1, y1, x2, y2, outline=colors[current_class], width=0, fill=colors[current_class])
            update_array(x1, y1, x2, y2, current_class)
    else:
        canvas.create_rectangle(x1, y1, x2, y2, outline="white", width=0, fill="white")
        clear_array(x1, y1, x2, y2)

# Function to store the initial mouse click coordinates
def on_click(event):
    global start_x, start_y
    start_x, start_y = event.x, event.y

# Function to change the current class
def change_class(index):
    global current_class
    current_class = index

# Function to update the NumPy array with a value of 1 if the cell is occupied
def update_array(x1, y1, x2, y2, class_index):
    global arr
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    filled_rectangle = np.full((y2 - y1, x2 - x1), 1)
    arr[class_index, y1:y2, x1:x2] = filled_rectangle

def update_array_wall(x1, y1, x2, y2, class_index):
    global arr
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    # filled_rectangle = np.full((y2 - y1, x2 - x1), 1)
    if y1 == y2:
        arr[class_index, y1:y1+scale, x1:x2] = 1
    else:
        arr[class_index, y1:y2, x1:x1+scale] = 1
def clear_array(x1, y1, x2, y2):
    global arr
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    filled_rectangle = np.full((y2 - y1, x2 - x1), 0)
    arr[:, y1:y2, x1:x2] = filled_rectangle


def print_array():
    global arr,h,w,stride
    # Capture the canvas and save it as an image
    canvas.update()  # Ensure the canvas is up-to-date
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()

    dir = os.path.join('samplelayouts',expname + '_'+ str(h) + '-' + str(w) + '-' + str(stride))
    os.makedirs(dir,exist_ok=True)
    ImageGrab.grab().crop((x, y, x1, y1)).save(os.path.join(dir,'layout.png'))

    # Save the numpy array to a file
    conditionmaps = MaxPool2d(kernel_size=scale)(torch.from_numpy(arr)).numpy().astype(np.uint8)


    views = get_views(h, w, window_size=32, stride=stride)
    hmax = 32 + (len(views) - 1) * stride
    wmax = 32 + (len(views[0]) - 1) * stride
    conditionmaps = np.pad(conditionmaps, pad_width=((0, 0), (0, hmax - h), (0, wmax - w)), mode='constant')

    for i in range(len(views)):
        for j in range(len(views[0])):
            x_start, x_end, z_start, z_end = views[i][j]
            layoutcrop = conditionmaps[:, x_start:x_end, z_start:z_end]
            np.save(os.path.join(dir, "{}.npy".format(str(i) + '_' + str(j))), layoutcrop)

    # Close the application
    root.destroy()

def get_views(panorama_height, panorama_width, window_size=64, stride=32):
    num_blocks_height = ceil((panorama_height - window_size) / stride) + 1
    num_blocks_width = ceil((panorama_width - window_size) / stride) + 1
    total_num_blocks = int(num_blocks_height * num_blocks_width)
    views = [[] for _ in range(num_blocks_height)]
    for i in range(num_blocks_height):
        for j in range(num_blocks_width):
            h_start = int(i  * stride)
            h_end = h_start + window_size
            w_start = int(j  * stride)
            w_end = w_start + window_size
            views[i].append((h_start, h_end, w_start, w_end))
    return views
# ...

parser = argparse.ArgumentParser()
parser.add_argument('--h', type=int, default=32, help='x scale in in decimeter')
parser.add_argument('--w', type=int, default=56, help='z scale in in decimeter')
parser.add_argument('--stride', type=int, default=24, help='stride in in decimeter')
parser.add_argument('--expname', type=str, default='draw')
args = parser.parse_args()
# Create main window
root = tk.Tk()
root.title("Layout Map Canvas")

# Create canvas to draw rectangles
h, w   = args.h, args.w # in decimeter, change the size of canvas
stride =  args.stride  # in decimeter
expname = args.expname
scale = 10
height, width = h*scale, w*scale
canvas = tk.Canvas(root, bg="white", width=width, height=height)
canvas.pack()

# Draw horizontal scale
for i in range(0, width):
    if i % (5*scale) == 0:  # Major scale lines
        canvas.create_line(i, 0, i, 10, width=2)
        canvas.create_text(i, 12, text=str(i//scale/10), font=("Arial", 8))
    else:  # Minor scale lines
        canvas.create_line(i, 0, i, 5, width=1)

# Draw vertical scale
for i in range(0, height):
    if i % (5*scale) == 0:  # Major scale lines
        canvas.create_line(0, i, 10, i, width=2)
        canvas.create_text(12, i, text=str(i//scale/10), font=("Arial", 8))
    else:  # Minor scale lines
        canvas.create_line(0, i, 5, i, width=1)

# Bind mouse events to the canvas
canvas.bind("<Button-1>", on_click)
canvas.bind("<ButtonRelease-1>", on_release)

colors = [
    "red", "blue", "green", "yellow", "orange",
    "purple", "brown", "cyan", "magenta"
]

class_names = ['wall', 'floor', 'bed', 'cabinet', 'chair', 'lighting', 'sofa', 'stool', 'table']

# Create frames, buttons, and labels for each class
frames = []
buttons = []
labels = []
for i, (color, class_name) in enumerate(zip(colors, class_names)):
    frame = tk.Frame(root)
    frame.pack(side="left", padx=5, pady=5)
    frames.append(frame)

    button = tk.Button(frame, bg=color, width=2, height=1, command=lambda index=i: change_class(index))
    button.pack(padx=5, pady=5)
    buttons.append(button)

    label = tk.Label(frame, text=class_name, bg="white")
    label.pack(padx=5, pady=5)
    labels.append(label)
# Add a button to print the final NumPy arrays
print_button = tk.Button(root, text="Print Arrays", command=print_array)
print_button.pack(side="left", padx=5, pady=5)

eraser_button = tk.Button(root, text="Eraser", command=enable_eraser)
eraser_button.pack(side="left", padx=5, pady=5)

# Initialize variables
start_x, start_y = 0, 0
current_class = 0

# Create a separate NumPy array for each class
arr = np.zeros((len(colors), height, width), dtype=int)

# Set main window size to a larger dimension while keeping the canvas size the same
root.geometry("1200x600")

# Run main loop
root.mainloop()