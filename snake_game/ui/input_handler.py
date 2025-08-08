# ui/input_handler.py
class InputHandler:
    def __init__(self, root, snake):
        self.root = root
        self.snake = snake
        self._bind_keys()

    def _bind_keys(self):
        self.root.bind("<Up>", lambda e: self._on_key((0, -1)))
        self.root.bind("<Down>", lambda e: self._on_key((0, 1)))
        self.root.bind("<Left>", lambda e: self._on_key((-1, 0)))
        self.root.bind("<Right>", lambda e: self._on_key((1, 0)))
        self.root.focus_set()

    def _on_key(self, direction):
        self.snake.set_direction(direction)
