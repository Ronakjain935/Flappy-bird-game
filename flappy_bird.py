# pyrefly: ignore [missing-import]
import gymnasium as gym
# pyrefly: ignore [missing-import]
import flappy_bird_gymnasium
# pyrefly: ignore [missing-import]
import pygame

# Creating our env
env = gym.make("FlappyBird-v0", render_mode="human")
state, info = env.reset()
# Initialize PyGame keyboard
pygame.init()
screen = pygame.display.get_surface()  # Gym has already created a window

terminated = False
truncated = False

while not (terminated or truncated):
    action = 0  # default -> 0 is no flap & 1 is flap

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            terminated = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                action = 1  # flap
                
    if terminated:
        break

    state, reward, terminated, truncated, info = env.step(action)
    env.render()

env.close()
pygame.quit()