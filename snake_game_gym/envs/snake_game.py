"""
Classic snake-game system implemented by Enzo Di Tizio.
"""

import gym
from gym import spaces, logger
import numpy as np
import random
import pygame


class SnakeGameEnv(gym.Env):
    """
    Description:
        A snake that start with 3 pieces length spawns in a map and can move into 3
        directions to get food. Each time that it gets one food, it grows 1 piece.
        The snake dies if it hits itself or the wall, and wins the game if occupy
        all the map.
    Source:
        This environment corresponds to the general worldwide played version of the
        snake-game.
    Observation:
        Type: MultiDiscrete([14] * 8 * 3 + [225])
        Num             Observation                     Min      Max
        0               Snake length                    0        225
        [1...8]         Apple Distance Ray Sensor       0        14
        [9...16]        Piece Distance Ray Sensor       0        14
        [17...24]       Wall Distance Ray Sensor        0        14
        Note 1: Ray distance is minus 1, since 0 means that it is in the adjacent tile.
        Note 2: Rays rotate in the global direction (Ref below) with the snake head.
    Actions:
        Type: Discrete(3)
        Num   Action
        0     Change head direction to right
        1     Keep same direction (do nothing)
        2     Change head direction to left
        Note: The left and right directions are related to the snake heads
    Reward:
        Reward is 1 for every step walked, and 10 for every apple taken
    Starting State:
        A normal state is calculated.
    Episode Termination:
        Snake head is inside any other snake piece.
        Snake is outside the field.
        Snake length is equals to 225.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        220.0 over 100 consecutive trials.
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}

    def __init__(self):
        self.x_size = 15
        self.y_size = 15

        # Actions and Observations
        self.action_space = spaces.Discrete(3)
        # Ray maximum length times ray quantity times ray channels plus the snake length
        self.observation_space = spaces.MultiDiscrete([15] * 8 * 3 + [226])

        self.window = None
        self.clock = None
        self.state = None
        self.snake = None
        self.apple = None
        self.global_snake_direction = None
        # Note: global_snake_direction refers to the global direction of the snake, not
        # equals as the action ones
        # Global:
        #         0
        #      3     1
        #         2

    def initial_parts(self):
        self.snake = [
            [7, 4],
            [7, 3],
        ]

        self.apple = [7, 11]

    def relative_to_global_direction(self, global_direction, relative_direction):
        # Turning right
        if relative_direction == 0:
            global_direction += 1
            if global_direction == 5:
                global_direction = 0

        # Turning left
        elif relative_direction == 2:
            global_direction -= 1
            if global_direction == -1:
                global_direction = 4

        return global_direction

    def move_snake(self):
        # Each piece except the head takes the position of the next piece
        for i in range(len(self.snake) - 1, 0, -1):
            # This is to the new piece after eating an apple got attached
            if self.snake[i] != self.snake[i - 1]:
                self.snake[i] = self.snake[i - 1].copy()

        # Move the head
        # If going up
        if self.global_snake_direction == 0:
            # Subtract on the y axis
            self.snake[0][0] -= 1
        # If going right
        elif self.global_snake_direction == 1:
            # Sum on the x axis
            self.snake[0][1] += 1
        # If going down
        elif self.global_snake_direction == 2:
            # Sum on the y axis
            self.snake[0][0] += 1
        # If going left
        else:
            # Subtract on the x axis
            self.snake[0][1] -= 1

    def generate_new_random_apple(self):
        self.apple = [random.randint(0, 14), random.randint(0, 14)]

    def check_if_ate_apple(self):
        for piece in self.snake:
            if piece == self.apple:
                self.snake.append(self.snake[-1].copy())
                self.generate_new_random_apple()

    def check_if_hit_wall(self):
        head_x = self.snake[0][0]
        head_y = self.snake[0][1]

        if head_x > 14:
            return True
        elif head_y > 14:
            return True
        elif head_x < 0:
            return True
        elif head_y < 0:
            return True
        else:
            return False

    def check_if_hit_itself(self):
        if self.snake[0] in self.snake[1:]:
            return True
        else:
            return False

    def check_if_won(self):
        if len(self.snake) == self.x_size * self.y_size:
            return True
        else:
            return False

    def apple_ray(self, obs, y_behaviour, x_behaviour):
        ray_position = self.snake[0].copy()
        for distance in range(15):
            ray_position[0] += y_behaviour
            ray_position[1] += x_behaviour
            if ray_position == self.apple:
                break
        obs.append(distance)

    def apple_ray_observation(self):
        obs = []

        self.apple_ray(obs, -1, 0)

        self.apple_ray(obs, -1, 1)

        self.apple_ray(obs, 0, 1)

        self.apple_ray(obs, 1, 1)

        self.apple_ray(obs, 1, 0)

        self.apple_ray(obs, 1, -1)

        self.apple_ray(obs, 0, -1)

        self.apple_ray(obs, -1, -1)

        return self.rotate_obs(obs)

    def wall_ray(self, obs, y_behaviour, x_behaviour):
        ray_position = self.snake[0].copy()
        for distance in range(15):
            ray_position[0] += y_behaviour
            ray_position[1] += x_behaviour
            if (
                ray_position[0] < 0
                or ray_position[0] > 14
                or ray_position[1] < 0
                or ray_position[1] > 14
            ):
                break
        obs.append(distance)

    def wall_ray_observation(self):
        obs = []

        self.wall_ray(obs, -1, 0)

        self.wall_ray(obs, -1, 1)

        self.wall_ray(obs, 0, 1)

        self.wall_ray(obs, 1, 1)

        self.wall_ray(obs, 1, 0)

        self.wall_ray(obs, 1, -1)

        self.wall_ray(obs, 0, -1)

        self.wall_ray(obs, -1, -1)

        return self.rotate_obs(obs)

    def snake_piece_ray(self, obs, y_behaviour, x_behaviour):
        ray_position = self.snake[0].copy()
        for distance in range(15):
            ray_position[0] += y_behaviour
            ray_position[1] += x_behaviour
            if ray_position in self.snake[1:]:
                break
        obs.append(distance)

    def snake_piece_ray_observation(self):
        obs = []

        self.snake_piece_ray(obs, -1, 0)

        self.snake_piece_ray(obs, -1, 1)

        self.snake_piece_ray(obs, 0, 1)

        self.snake_piece_ray(obs, 1, 1)

        self.snake_piece_ray(obs, 1, 0)

        self.snake_piece_ray(obs, 1, -1)

        self.snake_piece_ray(obs, 0, -1)

        self.snake_piece_ray(obs, -1, -1)

        return self.rotate_obs(obs)

    def rotate(self, l, n):
        """
        Function to do what's described in the rotate_obs fuction
        """
        return l[-n:] + l[:-n]

    def rotate_obs(self, obs):
        """
        Please notice that 90 deg means 2 rays clockwise, since each one is 45 deg
        - Up direction (numbers represent indexes):
                 0
              7     1
            6         2
              5     3
                 4
        - Right direction (up rotated 90 deg clockwise)
                 6
              5     7
            4         0
              3     1
                2
        Notice that 0 went 2 indexes to the right
        """

        # If going up, don't rotate, since its the default one
        if self.global_snake_direction == 0:
            return obs

        # If going right, rotate 90 deg (2 rays or indexes)
        elif self.global_snake_direction == 1:
            return self.rotate(obs, 2)

        # If going down, rotate 180 deg (4 rays or indexes)
        elif self.global_snake_direction == 2:
            return self.rotate(obs, 4)

        # If going left, rotate 270 deg (6 rays or indexes)
        else:
            return self.rotate(obs, 6)

    def generate_state(self):
        new_state = []

        new_state.append(len(self.snake))

        new_state += self.apple_ray_observation()

        new_state += self.wall_ray_observation()

        new_state += self.snake_piece_ray_observation()

        self.state = np.array(new_state)

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg

        self.global_snake_direction = self.relative_to_global_direction(
            self.global_snake_direction, action
        )

        self.move_snake()

        ate_apple = self.check_if_ate_apple()

        died = self.check_if_hit_wall() or self.check_if_hit_itself()

        won = self.check_if_won()

        self.generate_state()

        done = died or won

        if not done:
            if ate_apple:
                reward = 10.0
            else:
                reward = 1.0
        else:
            reward = 0.0

        if done:
            logger.warn(
                "You are calling 'step()' even though this "
                "environment has already returned done = True. You "
                "should always call 'reset()' once you receive 'done = "
                "True' -- any further steps are undefined behavior."
            )

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.initial_parts()
        self.global_snake_direction = 1
        self.generate_state()
        return self.state

    def render_background(self, screen_width, screen_height):
        self.canvas = pygame.Surface((screen_width, screen_height))
        self.canvas.fill((0, 0, 0))

    def render_snake_pieces(self, tile_size):
        for piece in self.snake:
            l, b, t, r = (
                piece[1] * tile_size,
                piece[0] * tile_size + tile_size,
                piece[0] * tile_size,
                piece[1] * tile_size + tile_size,
            )
            pygame.draw.rect(
                self.canvas,
                (0, 255, 0),
                pygame.Rect(l, t, r - l, b - t),
            )

    def render_apple(self, tile_size):
        l, b, t, r = (
            self.apple[1] * tile_size,
            self.apple[0] * tile_size + tile_size,
            self.apple[0] * tile_size,
            self.apple[1] * tile_size + tile_size,
        )
        pygame.draw.rect(
            self.canvas,
            (255, 0, 0),
            pygame.Rect(l, t, r - l, b - t),
        )

    def render(self, mode="human"):
        if self.state is None:
            return None

        tile_size = 40

        screen_width = tile_size * 15
        screen_height = tile_size * 15

        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((screen_width, screen_height))

        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        self.render_background(screen_width, screen_height)
        self.render_snake_pieces(tile_size)
        self.render_apple(tile_size)

        if mode == "human":
            self.window.blit(self.canvas, self.canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
