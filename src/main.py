import pygame
import pygame_gui
import sys
import classes

# Load models
models = {
    'stacking_classifier_fallecidos': classes.Model(),
    'stacking_classifier_graves': classes.Model(),
    'stacking_classifier_menos_graves': classes.Model(),
    'stacking_classifier_leves': classes.Model(),
    'stacking_classifier_ilesos': classes.Model(),
}
for model_name, model in models.items():
    model.load(f'models/{model_name}.pkl')

# Load columns
min_max_columns = classes.Columns()
min_max_columns.load('data/min_max_columns.csv')

# ---------------------------------------------------------------
# Initialize pygame
pygame.init()

# Create a window with a fixed size
window_surface = pygame.display.set_mode((800, 600))

manager = pygame_gui.UIManager((800, 600))

# Create a dropdown menu for model selection
model_selection = pygame_gui.elements.UIDropDownMenu(options_list=list(models.keys()), starting_option='stacking_classifier_fallecidos', relative_rect=pygame.Rect((250, 10), (300, 20)), manager=manager)

# Create 8 text entry boxes and labels
text_boxes = [pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((150, 60 + 60*i), (100, 50)), manager=manager) for i in range(8)]

# Create labels
labels = [pygame_gui.elements.UILabel(relative_rect=pygame.Rect((50, 60 + 60*i), (100, 50)), text=f'{name}', manager=manager) for i, name in enumerate(min_max_columns.indexes())]

min_and_max = [pygame_gui.elements.UILabel(relative_rect=pygame.Rect((240, 60 + 60*i), (100, 50)), text=f'{min_max_columns.min_and_max()[i]}', manager=manager) for i in range(8)]

# Create a button
button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((350, 550), (100, 50)), text='Predict', manager=manager)

# Create a label for the prediction
prediction_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((400, 200), (200, 50)), text='Prediction: ', manager=manager)

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
                    if classes.check_float([text_box.get_text() for text_box in text_boxes]):
                        inputs = [float(text_box.get_text()) for text_box in text_boxes]
                        inputs = min_max_columns.scaler([inputs])
                        print(inputs)
                        selected_model = models[model_selection.selected_option]
                        print('Predicción:', selected_model.predict([inputs])[0])
                        print('Probabilidades:', selected_model.predict_proba([inputs])[0])
                        # Update the prediction label
                        prediction = selected_model.predict([inputs])[0]
                        prediction_label.set_text(f'Prediction: {prediction}')

                    else:
                        prediction_label.set_text(f'Error: invalid input')

        manager.process_events(event)

    manager.update(time_delta)

    window_surface.fill((0, 107, 107))

    manager.draw_ui(window_surface)

    pygame.display.update()