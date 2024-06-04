#   Imports

import numpy as np
import pygame
import sys
import tensorflow as tf
import cv2
from PIL import Image

#   Check command-line arguments

if len(sys.argv) != 2:
    sys.exit('Usage: python recognition.py model.h5')

model = tf.keras.models.load_model(sys.argv[1])

#   Colors

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

#   Initialize Pygame

pygame.init()
size = width, height = 600, 400
screen = pygame.display.set_mode(size)

#   Fonts

OPEN_SANS = 'assets/fonts/OpenSans-Regular.ttf'
smallFont = pygame.font.Font(OPEN_SANS, 20)
largeFont = pygame.font.Font(OPEN_SANS, 40)

#   Board parameters

ROWS, COLS = 28, 28
OFFSET = 20
CELL_SIZE = 10

#   Image loading for test

img = cv2.imread('gtsrb/22/00001_00007.ppm')
img = cv2.resize(img, (30, 30))
pil_image = Image.open('gtsrb/22/00001_00007.ppm')
image_data = pil_image.tobytes()
image_dimensions = pil_image.size
pygame_surface = pygame.image.fromstring(image_data, image_dimensions, "RGB")
screen.blit(pygame_surface, (0, 0))
# pygame.display.flip()
classification = None

while True:

    #   Check if game quit

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    screen.fill(BLACK)

    #   Check for mouse click

    click, _, _ = pygame.mouse.get_pressed()
    
    if click == 1:
        mouse = pygame.mouse.get_pos()
    else:
        mouse = None
    

    #   Classify button
    classifyButton = pygame.Rect(
        150, OFFSET + ROWS * CELL_SIZE + 30,
        100, 30
    )
    classifyText = smallFont.render('Classify', True, BLACK)
    classifyTextRect = classifyText.get_rect()
    classifyTextRect.center = classifyButton.center
    pygame.draw.rect(screen, WHITE, classifyButton)
    screen.blit(classifyText, classifyTextRect)

    # #   Reset drawing
    # if mouse and resetButton.collidepoint(mouse):
    #     handwriting = [[0] * COLS for _ in range(ROWS)]
    #     classification = None

    #   Generate classification
    if mouse and classifyButton.collidepoint(mouse):
        classification = model.predict(
            [np.array(img).reshape(1, 30, 30, 3)]
        ).argmax()

    #   Show classification if one exists
    if classification is not None:
        classificationText = largeFont.render(str(classification), True, WHITE)
        classificationRect = classificationText.get_rect()
        grid_size = OFFSET * 2 + CELL_SIZE * COLS
        classificationRect.center = (
            grid_size + ((width - grid_size) / 2),
            100
        )
        screen.blit(classificationText, classificationRect)

    pygame.display.flip()