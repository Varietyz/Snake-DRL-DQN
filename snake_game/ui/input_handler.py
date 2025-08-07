# snake_game/ui/input_handler.py
class InputHandler:
    def __init__(self, root, snake, ai_toggle_callback, ai_enabled_getter):
        self.root = root
        self.snake = snake
        self._ai_toggle_callback = ai_toggle_callback
        self._ai_enabled_getter = ai_enabled_getter
        self._bind_keys()

    def _bind_keys(self):
        self.root.bind("<Up>", lambda e: self._on_key((0, -1)))
        self.root.bind("<Down>", lambda e: self._on_key((0, 1)))
        self.root.bind("<Left>", lambda e: self._on_key((-1, 0)))
        self.root.bind("<Right>", lambda e: self._on_key((1, 0)))
        self.root.bind("<space>", self._toggle_ai)
        self.root.focus_set()

    def _on_key(self, direction):
        if not self._ai_enabled_getter():
            self.snake.set_direction(direction)

    def _toggle_ai(self, event=None):
        self._ai_toggle_callback()