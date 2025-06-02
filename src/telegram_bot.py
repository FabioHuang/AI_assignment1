import os
import logging
import numpy as np
from telegram import Update, ParseMode
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

# Assuming model.py and face_extractor.py are in the same directory or accessible via PYTHONPATH
from face_recognition.model import KNNFaceClassifier
from face_recognition.preprocessing import get_image_embeddings

# --- Configuration ---
TELEGRAM_TOKEN = ""
MODEL_PATH = "/home/fabio/Repos/AI_assignment1/models/knn_face_classifier.pkl"

UNKNOWN_THRESHOLD = 0.3

EMBEDDING_MODEL_NAME = "ArcFace"

# --- Logging Setup ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Load Model ---
try:
    face_classifier = KNNFaceClassifier.load(MODEL_PATH)
    face_classifier.set_unknown_threshold(UNKNOWN_THRESHOLD)
    logger.info(f"KNN Face Classifier loaded successfully from {MODEL_PATH}")
    logger.info(f"Using unknown threshold: {UNKNOWN_THRESHOLD}, k: {face_classifier.best_k}, metric: {face_classifier.best_metric}")
except FileNotFoundError:
    logger.error(f"Model file not found at {MODEL_PATH}. Please ensure the model is trained and the path is correct.")
    exit()
except Exception as e:
    logger.error(f"Error loading KNN model: {e}", exc_info=True)
    exit()

# --- Bot Handlers ---
def start_command(update: Update, context: CallbackContext) -> None:
    """Sends a welcome message when the /start command is issued."""
    user = update.effective_user
    update.message.reply_html(
        rf"Hi {user.mention_html()}! I am the Doorman Bot 🤖.",
        reply_markup=None,
    )
    update.message.reply_text("Send me a photo of a person, and I'll try to recognize them!")

def process_photo(update: Update, context: CallbackContext) -> None:
    """Processes a photo sent by the user to recognize a face."""
    if not update.message.photo:
        update.message.reply_text("Please send a photo.")
        return

    chat_id = update.effective_chat.id
    message_id = update.message.message_id

    # Get the largest photo sent
    photo_file = update.message.photo[-1].get_file()
    
    # Temporary path to save the downloaded photo
    temp_photo_path = f"temp_image_{chat_id}_{message_id}.jpg"
    
    try:
        context.bot.send_chat_action(chat_id=chat_id, action="typing")
        photo_file.download(temp_photo_path)
        logger.info(f"Photo downloaded to {temp_photo_path} from user {update.effective_user.username}")

        # Extract face embedding
        embedding = get_image_embeddings(temp_photo_path, model=EMBEDDING_MODEL_NAME)

        if embedding is None:
            update.message.reply_text("I couldn't detect a face in the photo, or the face was not clear enough. Please try another one.")
            return

        embedding_reshaped = np.array(embedding).reshape(1, -1)
        predicted_name = face_classifier.predict(embedding_reshaped)[0]
        
        if predicted_name.lower() == "desconhecido":
            reply_text = f"Access denied."
        else:
            reply_text = f"✅ Hello, {predicted_name}! Access granted."
            
        update.message.reply_text(reply_text)

    except Exception as e:
        logger.error(f"Error processing photo: {e}", exc_info=True)
        update.message.reply_text("Sorry, an error occurred while processing the image. Please try again.")
    finally:
        # Clean up the temporary photo file
        if os.path.exists(temp_photo_path):
            os.remove(temp_photo_path)

def error_handler(update: Update, context: CallbackContext) -> None:
    """Log Errors caused by Updates."""
    logger.warning(f'Update "{update}" caused error "{context.error}"')


def main() -> None:
    if TELEGRAM_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN":
        logger.error("CRITICAL: TELEGRAM_BOT_TOKEN is not set. Please set it in the script or as an environment variable.")
        return

    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("start", start_command))

    dispatcher.add_handler(MessageHandler(Filters.photo & ~Filters.command, process_photo))
    
    dispatcher.add_error_handler(error_handler)

    logger.info("Starting Doorman Bot...")
    updater.start_polling()
    logger.info("Bot is running. Press Ctrl-C to stop.")
    updater.idle()

if __name__ == '__main__':
    main()
