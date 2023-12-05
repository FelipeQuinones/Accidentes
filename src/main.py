import pygame
import pygame_gui
import sys
import classes

# Load models
model_fallecidos = classes.Model()
model_fallecidos.load('models/stacking_classifier_fallecidos.pkl')

# Load columns
min_max_columns = classes.Columns()
min_max_columns.load('data/min_max_columns.csv')

# Initialize pygame
pygame.init()

# Create a window
window_surface = pygame.display.set_mode((800, 600))

manager = pygame_gui.UIManager((800, 600))

# Create 8 text entry boxes and labels
text_boxes = [pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((350, 50 + 60*i), (100, 50)), manager=manager) for i in range(8)]
labels = [pygame_gui.elements.UILabel(relative_rect=pygame.Rect((460, 50 + 60*i), (100, 50)), text=f'{name}', manager=manager) for i, name in enumerate(min_max_columns.indexes())]

# Create a button
button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((350, 550), (100, 50)), text='Save', manager=manager)

inputs = []

clock = pygame.time.Clock()

# Main loop
while True:
    time_delta = clock.tick(60)/1000.0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.USEREVENT:
            if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == button:
                    inputs = [float(text_box.get_text()) for text_box in text_boxes]
                    inputs = min_max_columns.scaler([inputs])
                    print(inputs)
                    print('Predicción:', model_fallecidos.predict([inputs])[0])
                    

        manager.process_events(event)

    manager.update(time_delta)

    window_surface.fill((255, 255, 255))

    manager.draw_ui(window_surface)

    pygame.display.update()