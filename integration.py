import cv2
import pytesseract
import matplotlib.pyplot as plt 
import pandas as pd
import os
from tkinter import Tk, StringVar, Scrollbar, Canvas, Frame, Button, END
from tkinter import ttk  # Import ttk module for better styling
from tkcalendar import DateEntry  # Import DateEntry for date selection
# from tensorflow import keras
import numpy as np

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Load the image
img = cv2.imread('6.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur
blur = cv2.GaussianBlur(gray, (5,5), 0)

# Apply Otsu's binarization
_, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Dilate the image
kernel = np.ones((2,2), np.uint8)
dilated = cv2.dilate(binary, kernel, iterations=2)

# Detect the characters
hImg, wImg = dilated.shape
cong = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789,'
text = pytesseract.image_to_string(dilated, config=cong)

# Split the extracted text by commas to get the roll numbers
roll_numbers = [roll.strip() for roll in text.replace('\n', ',').split(',')]
# Print the extracted roll numbers
for roll in roll_numbers:
    print("Extracted Roll Number:", roll)

# Create a Tkinter window
root = Tk()

# Create a frame for the checkboxes
frame = Frame(root)
frame.pack()

# Create a scrollbar
scrollbar = Scrollbar(frame)
scrollbar.pack(side='right', fill='y')

# Create a canvas
canvas = Canvas(frame, yscrollcommand=scrollbar.set)
canvas.pack(side='left')

# Add the checkboxes to the canvas
checkboxes = []
for roll in roll_numbers:
    var = StringVar(value=roll)
    checkbox = ttk.Checkbutton(canvas, text="Roll Number: " + roll, variable=var, onvalue=roll, offvalue='')  # Use ttk.Checkbutton
    checkbox.var = var 
    checkbox.pack()
    checkboxes.append(checkbox)

scrollbar.config(command=canvas.yview)

# Add a DateEntry widget for date selection
date_entry = DateEntry(root, width=12, year=2022, month=1, day=1)
date_entry.pack(padx=10, pady=10)

def confirm():
    confirmed_roll_numbers = [cb.var.get() for cb in checkboxes if cb.var.get()]
    confirmed_roll_numbers = [roll for rolls in confirmed_roll_numbers for roll in rolls.split('\n')]
    print(confirmed_roll_numbers)
    # Get the selected date
    selected_date = date_entry.get_date().strftime('%Y-%m-%d')

    # Check if the file exists
    if os.path.isfile('roll_numbers.xlsx'):
        # If the file exists, read it into a DataFrame
        df = pd.read_excel('roll_numbers.xlsx')
    else:
        # If the file doesn't exist, create a new DataFrame with roll numbers from 1 to 100
        df = pd.DataFrame({'Roll Number': range(1, 101)})

    # Convert all columns to object dtype
    df = df.astype(object)

    # For each roll number in the image, put 'P' in the corresponding cell
    # For each roll number in the image, put 'P' in the corresponding cell
    for roll in confirmed_roll_numbers:
        if selected_date not in df.columns:
            df[selected_date] = np.nan
        df[selected_date] = df[selected_date].astype(object)
        df.loc[df['Roll Number'] == int(roll), selected_date] = 'P'

     # Sort the DataFrame by column names (dates)
    df.set_index('Roll Number', inplace=True)
    df = df.sort_index(axis=1)
    df.reset_index(inplace=True)

    # Write the DataFrame to an Excel file
    df.to_excel('roll_numbers.xlsx', index=False)

    root.quit()

# Add a confirm button
confirm_button = ttk.Button(root, text="Confirm", command=confirm)  # Use ttk.Button
confirm_button.pack()

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Result')
plt.show()
# Run the Tkinter event loop
root.mainloop()