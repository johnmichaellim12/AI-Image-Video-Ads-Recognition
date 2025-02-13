import torch
import cv2
import os
import numpy as np
from ultralytics import YOLO
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch import nn, optim
from nudenet import NudeClassifier



# Load YOLOv8 model (pretrained on COCO dataset)
yolo_model = YOLO("yolov8n.pt")

# Load CLIP model (for text-image matching)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load BLIP model (for image captioning)
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load NudeNet classifier
nsfw_classifier = NudeClassifier()

# # # Load NSFW detection model
# nsfw_model = AutoModelForImageClassification.from_pretrained("openai/clip-vit-base-patch32")
# nsfw_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load NSFW detection model
# nsfw_model = predict.load_model("nsfw_mobilenet_v2.pb")  # Using a dedicated NSFW detection model

# Define NSFW labels (adjust based on dataset used for training CLIP)
NSFW_LABELS = ["porn", "sexy", "nudity", "explicit"]
SAFE_LABELS = ["safe", "clothing", "clean"]
SIMILARITY_THRESHOLD = 0.5  # Adjustable threshold for NSFW detection

def is_nsfw(image_path):
    """Use CLIP and NudeNet to classify NSFW content."""
    # CLIP NSFW Detection
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(text=NSFW_LABELS + SAFE_LABELS, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_model(**inputs)
    similarities = outputs.logits_per_image[0].softmax(dim=0)
    nsfw_score = sum(similarities[:len(NSFW_LABELS)])  # NSFW labels
    safe_score = sum(similarities[len(NSFW_LABELS):])  # Safe labels

    # NudeNet Classification
    nudenet_results = nsfw_classifier.classify(image_path)
    nudenet_nsfw_prob = nudenet_results[image_path].get('porn', 0) + nudenet_results[image_path].get('sexy', 0)

    # Final Decision: Require both CLIP and NudeNet to be over threshold
    if nsfw_score > SIMILARITY_THRESHOLD and nudenet_nsfw_prob > 0.5:
        print(f"🚨 NSFW detected in {image_path} (CLIP: {nsfw_score:.2f}, NudeNet: {nudenet_nsfw_prob:.2f})")
        return True

    return False

def move_nsfw_file(frame_path, frame_number):
    """Move NSFW file to the NSFW folder with a unique name if it already exists."""
    base_name, ext = os.path.splitext(os.path.basename(frame_path))
    new_path = os.path.join("output/nsfw", f"{base_name}_frame_{frame_number}{ext}")
    
    # Ensure the filename is unique
    counter = 1
    while os.path.exists(new_path):
        new_path = os.path.join("output/nsfw", f"{base_name}_frame_{frame_number}_{counter}{ext}")
        counter += 1

    os.rename(frame_path, new_path)
    print(f"🚨 NSFW file moved to {new_path}")

# # # NSFW Detection Function
# def is_nsfw(image_path):
#     """Detect if an image contains NSFW content."""
#     image = Image.open(image_path).convert("RGB")
#     inputs = nsfw_processor(images=image, return_tensors="pt")
#     outputs = nsfw_model(**inputs)
#     probs = outputs.logits.softmax(dim=-1).detach().numpy()[0]
#     nsfw_score = probs[1]  # Assuming index 1 is NSFW probability
#     return nsfw_score > 0.5  # Flag as NSFW if above 50%

# def is_nsfw(image_path):
#     """Detect if an image contains NSFW content."""
    
#     # Step 1: NudeNet Classification
#     results = nsfw_classifier.classify(image_path)
#     nsfw_prob = results[image_path].get('porn', 0) + results[image_path].get('sexy', 0)

#     # Step 2: DeepStack NSFW (if available)
#     try:
#         response = deepstack_detector.detectObject(image_path, min_confidence=0.6)
#         ds_nsfw = any(obj['label'] == 'nudity' for obj in response["predictions"])
#     except Exception:
#         ds_nsfw = False  # If DeepStack isn't running, ignore it.

#     # Step 3: Decision Based on Threshold
#     if nsfw_prob > 0.5 or ds_nsfw:
#         print(f"🚨 NSFW content detected in {image_path} (Score: {nsfw_prob:.2f})")
#         return True
    
#     return False

# Define dataset class
class NSFWImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root_dir, transform=transform)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, torch.tensor(label, dtype=torch.long)
    

# # Define training function
# def train_nsfw_model(data_dir, epochs=5, batch_size=16, lr=0.001):
#     """Fine-tune the NSFW classification model."""
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor()
#     ])
    
#     dataset = NSFWImageDataset(data_dir, transform=transform)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
#     model = AutoModelForImageClassification.from_pretrained("openai/clip-vit-base-patch32")
#     model.classifier = nn.Linear(model.config.hidden_size, 2)  # Adjust for binary classification
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)
    
#     model.train()
#     for epoch in range(epochs):
#         total_loss = 0.0
#         for images, labels in dataloader:
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs.logits, labels)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
#     # Save trained model
#     model.save_pretrained("nsfw_trained_model")
#     print("✅ Model training completed and saved.")

# Example usage for training
# Uncomment this to train the NSFW model with your dataset
# train_nsfw_model("dataset", epochs=5)

# Create output directories
def setup_output_folders():
    os.makedirs("output/approved", exist_ok=True)
    os.makedirs("output/rejected", exist_ok=True)
    os.makedirs("output/nsfw", exist_ok=True)
    os.makedirs("output/frames", exist_ok=True)

# Load allowed categories from a text file
def load_categories_from_file(filepath):
    """Load categories from a text file where each line is a category."""
    categories = {}
    if not os.path.exists(filepath):
        print(f"⚠️ Category file '{filepath}' not found. Please create it with one category per line.")
        return categories
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            category = line.strip().lower()
            if category:
                categories[category] = category  # Use category name as description for now
    if not categories:
        print(f"⚠️ No valid categories found in '{filepath}'. Please check the formatting.")
    return categories

# Load categories from 'categories.txt'
CATEGORY_FILE = "categories.txt"
ALLOWED_CATEGORIES = load_categories_from_file(CATEGORY_FILE)
if not ALLOWED_CATEGORIES:
    exit("❌ No categories loaded. Exiting script.")

SIMILARITY_THRESHOLD = 0.5  # Adjustable threshold for relevance

def prompt_user_for_review(image_path, caption, category):
    """Ask the user if an ad with a low similarity score matches the category."""
    user_input = input(f"❓ Does this ad match the category '{category}'? (y/n): {caption}\n")
    return user_input.lower() == 'y'

# def is_relevant_ad(caption, category):
#     """Check if the generated caption is relevant to the chosen category using similarity scoring."""
#     if category.lower() not in ALLOWED_CATEGORIES:
#         print("⚠️ Invalid category provided. Available categories:", list(ALLOWED_CATEGORIES.keys()))
#         return False
    
#     category_description = ALLOWED_CATEGORIES[category.lower()]
#     similarity_score = compute_similarity(caption, category_description)
    
#     print(f"🔎 Similarity Score ({category}): {similarity_score:.2f}")
    
#     if similarity_score < SIMILARITY_THRESHOLD:
#         return prompt_user_for_review(caption, category)
    
#     return similarity_score >= SIMILARITY_THRESHOLD


def detect_objects(image_path, output_folder):
    """Detect objects in an image using YOLOv8 and save output."""
    results = yolo_model(image_path)
    detections = results[0].boxes.data.cpu().numpy()
    
    image = cv2.imread(image_path)
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        label = yolo_model.names[int(cls)]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2, int(y2))), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, image)
    return detections

def generate_image_caption(image_path):
    """Generate a text description for an image using BLIP."""
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(images=image, return_tensors="pt")
    caption_ids = blip_model.generate(**inputs)
    caption = blip_processor.decode(caption_ids[0], skip_special_tokens=True)
    return caption

def compute_similarity(caption, category_description):
    """Calculate similarity between AI-generated caption and category description."""
    inputs = clip_processor(text=[category_description], images=Image.new('RGB', (224, 224)), return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    similarity = outputs.logits_per_image.item()
    return similarity

def is_relevant_ad(caption, category):
    """Check if the generated caption is relevant to the chosen category using similarity scoring."""
    if category.lower() not in ALLOWED_CATEGORIES:
        print("⚠️ Invalid category provided. Available categories:", list(ALLOWED_CATEGORIES.keys()))
        return False
    
    category_description = ALLOWED_CATEGORIES[category.lower()]
    similarity_score = compute_similarity(caption, category_description)
    
    print(f"🔎 Similarity Score ({category}): {similarity_score:.2f}")
    return similarity_score >= SIMILARITY_THRESHOLD

def process_video(video_path, category):
    """Extract frames from video and analyze content frame by frame."""
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Default to 30 FPS if unknown

    print(f"📹 Processing {video_path} - Total frames: {total_frames}, FPS: {fps}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop if no frame is read

        frame_path = os.path.join("output/frames", f"{os.path.basename(video_path)}_frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)

        # 🔍 Check for NSFW content
        if is_nsfw(frame_path):
            print(f"🚨 NSFW content detected in {video_path} - Frame {frame_count}")
            move_nsfw_file(frame_path, frame_count)
        
        frame_count += 1  # Process **every** frame (not just every 30th)

    cap.release()
    print(f"✅ {video_path} - Processing complete.")




def process_folder(folder_path, category):
    """Process all images and videos in a folder and organize outputs."""
    setup_output_folders()
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"Processing image: {filename}")
            if is_nsfw(file_path):
                print(f"🚨 NSFW content detected in {filename}")
                os.rename(file_path, os.path.join("output/nsfw", filename))
                continue  # Skip further processing
            text_prompt = generate_image_caption(file_path)
            if is_relevant_ad(text_prompt, category):
                output_folder = "output/approved"
                print(f"✅ {filename} - Ad approved under '{category}': {text_prompt}")
            else:
                output_folder = "output/rejected"
                print(f"❌ {filename} - Ad rejected under '{category}': {text_prompt}")
            detect_objects(file_path, output_folder)
        elif filename.lower().endswith(('.mp4', '.avi', '.mov', '.webm')):
            print(f"📹 Processing video: {filename}")
            process_video(file_path, category)
        else:
            print(f"⚠️ Skipping unsupported file: {filename}")

# Example usage
folder_path = input("Enter the folder path containing images/videos: ")
category = input(f"Enter the category: ")
process_folder(folder_path, category)
