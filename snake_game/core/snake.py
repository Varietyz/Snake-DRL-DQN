# core/snake.py
from collections import deque

class Snake:
    def __init__(self, init_pos, init_length=3):
        self.body = deque([init_pos])
        self.direction = (1, 0)
        for _ in range(1, init_length):
            self.body.appendleft((self.body[0][0] - 1, self.body[0][1]))
        self.grow_next = False

    def move(self):
        head_x, head_y = self.body[-1]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)
        self.body.append(new_head)
        if not self.grow_next:
            self.body.popleft()
        else:
            self.grow_next = False

    def grow(self):
        self.grow_next = True

    def set_direction(self, new_dir):
        if (new_dir[0] * -1, new_dir[1] * -1) == self.direction:
            return

        head_x, head_y = self.body[-1]
        dx, dy = new_dir
        new_head = (head_x + dx, head_y + dy)

        tail = self.body[0]
        if self.grow_next and new_head == tail:
            return

        self.direction = new_dir


    def hits_self(self):
        body_list = list(self.body)
        if not self.grow_next:
            body_list = body_list[1:]
        has_collision = len(body_list) != len(set(body_list))
        return has_collision



    def get_head(self):
        return self.body[-1]
