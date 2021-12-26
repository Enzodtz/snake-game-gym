# snake-game-gym

Use the popular snake game into open AI gym environments

## Installation

```bash
git clone https://github.com/Enzodtz/snake-game-gym.git
cd snake-game-gym
python3 setup.py install
```

## Usage

```python
import gym
import snake_game_gym

env = gym.make("snake-game-v0")
observation = env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

    if done:
      observation = env.reset()

env.close()
```
