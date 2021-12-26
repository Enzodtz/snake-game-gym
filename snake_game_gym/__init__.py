from gym.envs.registration import register

register(
    id="snake-game-v0",
    entry_point="snake_game_gym.envs:SnakeGameEnv",
)
