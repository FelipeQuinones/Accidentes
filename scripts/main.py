import pygame
import pygame_gui
import sys

pygame.init()

# Create a window
window_surface = pygame.display.set_mode((800, 600))

manager = pygame_gui.UIManager((800, 600))

# Create 8 text entry boxes
text_boxes = [pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((350, 50 + 60*i), (100, 50)), manager=manager) for i in range(8)]

clock = pygame.time.Clock()

# Main loop
while True:
    time_delta = clock.tick(60)/1000.0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        manager.process_events(event)

    manager.update(time_delta)

    window_surface.fill((255, 255, 255))

    manager.draw_ui(window_surface)

    pygame.display.update()
