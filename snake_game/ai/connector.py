# ai/connector.py
class Connector:
    ACTION_TO_DIRECTION = {
        0: (1, 0),    # right
        1: (-1, 0),   # left
        2: (0, 1),    # down
        3: (0, -1)    # up
    }

    def __init__(self, game_controller, agent_plugin):
        self.game_controller = game_controller
        self.agent = agent_plugin

    def get_game_state(self):
        snake_body = list(self.game_controller.snake.body)
        food_pos = self.game_controller.food
        bonus_food_pos = self.game_controller.bonus_manager.bonus_food

        return {
            "snake_body": snake_body,
            "food_pos": food_pos,
            "bonus_food_pos": bonus_food_pos,
            "score": self.game_controller.score.current_score,
            # Add other state info if needed
        }

    def request_action(self):
        # For example, pass full controller:
        action = self.agent.act(self.game_controller)
        return action

    def apply_action(self, action):
        if action is not None:
            direction = self.ACTION_TO_DIRECTION.get(action)
            if direction is not None:
                self.game_controller.snake.set_direction(direction)

    def update(self):
        action = self.request_action()
        self.apply_action(action)
