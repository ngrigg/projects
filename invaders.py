import pygame
import random

# Initialize Pygame
pygame.init()

# Screen dimensions
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))

# Colors
black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)

# Game title
pygame.display.set_caption("Space Invaders")

# Player attributes
player_width = 60
player_height = 10
player_x = (screen_width * 0.45)
player_y = (screen_height - player_height - 10)
player_speed = 5

# Enemy attributes
enemy_width = 60
enemy_height = 10
enemy_speed = 2
enemy_x_change = enemy_speed
enemy_y_change = enemy_height
enemies = []

for i in range(6):  # Create 6 enemies
    enemies.append([random.randint(0, screen_width-enemy_width), random.randint(0, 100)])

# Main game loop
running = True
while running:
    screen.fill(black)  # Clear the screen

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Move the player
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        player_x -= player_speed
        if player_x < 0:
            player_x = 0
    if keys[pygame.K_RIGHT]:
        player_x += player_speed
        if player_x > screen_width - player_width:
            player_x = screen_width - player_width

    # Draw the player
    pygame.draw.rect(screen, white, [player_x, player_y, player_width, player_height])

    # Move and draw the enemies
    for enemy in enemies:
        enemy[0] += enemy_x_change
        if enemy[0] <= 0 or enemy[0] >= screen_width - enemy_width:
            enemy_x_change = -enemy_x_change
            enemy[1] += enemy_y_change
        pygame.draw.rect(screen, red, [enemy[0], enemy[1], enemy_width, enemy_height])

    pygame.display.flip()  # Update the screen

    # Cap the frame rate
    pygame.time.Clock().tick(60)

pygame.quit()
