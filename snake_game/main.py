# main.py
import tkinter as tk
from config import GRID_WIDTH, GRID_HEIGHT, GRID_SIZE, COLORS
from ui.controller import GameController

def main():
    root = tk.Tk()
    root.title("Modular Snake Game")
    canvas = tk.Canvas(
        root,
        width=GRID_WIDTH * GRID_SIZE,
        height=GRID_HEIGHT * GRID_SIZE,
        bg=COLORS["background"]
    )
    canvas.pack()
    GameController(root, canvas)
    root.mainloop()

if __name__ == "__main__":
    main()
