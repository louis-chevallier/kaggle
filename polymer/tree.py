
import torch
import math
import tkinter as tk
import time
import threading
from utillc import *
from PIL import Image, ImageTk
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
K = 1

class Threshold(torch.nn.Module):
	def __init__(self, shape):
		super().__init__()
		self.threshold = torch.nn.Parameter(torch.randn(shape))

	def forward(self, x):
		d = (self.x - self.threshold) * K
		return
_range = 1
start, end = -3, 3

# Créer et lancer la fenêtre
root = tk.Tk()
root.title("sigmoid")
root.geometry("800x800")
label = tk.Label(root, text="sigmoid", font=('Arial', 14))
label.pack(pady=20)

mycanvas = tk.Canvas(root,
					 width=850, height=400,
					 bg="white", highlightthickness=0)
image = mycanvas.create_image(0, 0, anchor='center')
#_ = tk.Label(root, text="xx")

def cb(v=0) :
	#EKO()
	x = torch.linspace(start, end, 100)
	m = torch.nn.Sigmoid()
	y = m(x*float(v))
	fig = plt.figure()
	ax = fig.gca()
	ax.set_xticks(np.arange(start, end, 1))
	plt.plot(x, y)
	plt.grid()
	# Save figure in PNG format to byte stream
	b = BytesIO()
	fig.savefig(b, format='png')
	
	# Read back from byte stream
	b.seek(0)
	img = plt.imread(b)*255
	h, w, depth = img.shape
	PIL_image = Image.fromarray(img.astype('uint8'), 'RGBA')
	PIL_image.save("toto.png")
	b.close()
	del b
	plt.close()
	photo = ImageTk.PhotoImage(image=PIL_image)
	#photo = ImageTk.PhotoImage(img)
	#photo = self.photo_image(np.hstack((img1, img2)))
	mycanvas.create_image(0, 0, image=photo, anchor=tk.NW)	
	#mycanvas.itemconfig(image, image=PIL_image)

	root.geometry('%03dx%03d' % (w, h + 180))
	#mycanvas.pack(fill=tk.BOTH, expand=tk.YES)
	EKO()
	
w = tk.Scale(root, from_=0, to=_range*50, resolution=0.01,
			 orient=tk.HORIZONTAL,
			 command=cb)


mycanvas.pack(fill=tk.BOTH, expand=tk.YES)
w.pack(anchor=tk.CENTER, fill=tk.BOTH, expand=tk.YES)

root.mainloop()
	
		
