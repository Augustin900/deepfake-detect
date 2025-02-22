from requirements import *
from settings import *
from common_functions import *

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
    print_error_header("No images have been found in the following folders: ")

    if not train_real_list:
        print_error_ext(f"{train_real_dir}")

    if not train_fake_list:
        print_error_ext(f"{train_fake_dir}")

    if not val_real_list:
        print_error_ext(f"{val_real_dir}")
        
    if not val_fake_list:
        print_error_ext(f"{val_real_dir}")

    print_error_end()

print("|- ======================================================================== -|")

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
save_model(model, MODEL_SAVE_PATH)