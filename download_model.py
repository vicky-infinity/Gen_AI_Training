from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_PATH = "./models/all-MiniLM-L6-v2"

print("Downloading model...")

model = SentenceTransformer(MODEL_NAME)

print("Saving model locally...")

model.save(MODEL_PATH)

print(f"Model saved successfully at: {MODEL_PATH}")