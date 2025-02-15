from requirements import *
from settings import *

def insert_at_position(text, insert_chars, position):
    """
    Inserts a set of characters at a specific position in the text, adjusting the text to ensure
    the characters are always at that position.

    Parameters:
    - text (str): The original text.
    - insert_chars (str): The characters to insert.
    - position (int): The position at which to insert the characters (0-indexed).

    Returns:
    - str: The modified text with the inserted characters.
    """
    # Ensure the position is within bounds
    if position < 0:
        position = 0

    # Create padding if needed
    if len(text) < position:
        text = text.ljust(position)

    # Insert the characters
    modified_text = text[:position] + insert_chars + text[position:]
    
    return modified_text

def insert_at_position_and_print(text, insert_chars, position):
    print(insert_at_position(text, insert_chars, position))

def print_error(message, quit=True):
    print(f"{fg('red')}", end="")
    insert_at_position_and_print(f"|- {"E: " if not quit else ""}{message}{"Aborting!" if quit else ""}", " -|", 75)
    print(f"{attr('reset')}", end="")

    if quit:
        print("|- ======================================================================== -|")
        exit()

def print_error_ext(message):
    print(f"{fg('red')}", end="")
    insert_at_position_and_print(f"|- {message}", " -|", 75)
    print(f"{attr('reset')}", end="")

def print_info(message):
    print(f"{fg('blue')}", end="")
    insert_at_position_and_print(f"|- I: {message}", " -|", 75)
    print(f"{attr('reset')}", end="")

def print_info_ext(message):
    print(f"{fg('blue')}", end="")
    insert_at_position_and_print(f"|- {message}", " -|", 75)
    print(f"{attr('reset')}", end="")

system("cls" if name == "nt" else "clear")

print("|- ======================================================================== -|")

print_info("Train data directories: ")
print_info_ext(f"{listdir(TRAIN_DIR)}")
print_info("Validation data directories: ")
print_info_ext(f"{listdir(VAL_DIR)}")

# Check if there are image files in the 'real' and 'fake' folders
train_real_dir = join(TRAIN_DIR, 'real')
train_real_list = listdir(train_real_dir)
train_fake_dir = join(TRAIN_DIR, 'fake')
train_fake_list = listdir(train_fake_dir)
val_real_dir = join(VAL_DIR, 'real')
val_real_list = listdir(train_real_dir)
val_fake_dir = join(VAL_DIR, 'fake')
val_fake_list = listdir(train_fake_dir)

print_info("Training images: ")
print_info_ext(f"Real: {train_real_list}")
print_info_ext(f"Fake: {train_fake_list}")
print_info("Evaluation images: ")
print_info_ext(f"Real: {val_real_list}")
print_info_ext(f"Fake: {val_fake_list}")

if not train_real_list or not train_fake_list or not val_real_list or not val_fake_list:
    print_error(f"No images have been found in the following folders: ", False)

    if not train_real_list:
        print_error_ext(f"{train_real_dir}")

    if not train_fake_list:
        print_error_ext(f"{train_fake_dir}")

    if not val_real_list:
        print_error_ext(f"{val_real_dir}")
        
    if not val_fake_list:
        print_error_ext(f"{val_real_dir}")

    print_error(f"", True)

# Data Augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Data preparation for validation set (no augmentation)
val_datagen = ImageDataGenerator(rescale=1./255)

# Load training and validation data
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'  # 'binary' for real vs. fake classification
)

validation_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Load pre-trained EfficientNetB0 model and freeze its layers
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
base_model.trainable = False

# Create a new model with a custom head for deepfake detection
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks: Save the best model and stop early if no improvement
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=[checkpoint, early_stopping]
)

# Optionally, save the final model
model.save(MODEL_SAVE_PATH)
