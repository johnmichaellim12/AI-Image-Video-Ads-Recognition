import torch
import cv2
import os
import numpy as np
from ultralytics import YOLO
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration, AutoModelForImageClassification, AutoProcessor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch import nn, optim
from nsfw_detector import predict

# Load YOLOv8 model (pretrained on COCO dataset)
yolo_model = YOLO("yolov8n.pt")

# Load CLIP model (for text-image matching)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load BLIP model (for image captioning)
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# # Load NSFW detection model
# nsfw_model = AutoModelForImageClassification.from_pretrained("openai/clip-vit-base-patch32")
# nsfw_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load NSFW detection model
nsfw_model = predict.load_model("nsfw_mobilenet_v2.pb")  # Using a dedicated NSFW detection model

def move_nsfw_file(frame_path):
    """Move NSFW file only if it doesn't already exist in the NSFW folder."""
    new_path = os.path.join("output/nsfw", os.path.basename(frame_path))
    if not os.path.exists(new_path):
        os.rename(frame_path, new_path)
    else:
        print(f"⚠️ NSFW file already exists: {new_path}, skipping rename.")

# # NSFW Detection Function
# def is_nsfw(image_path):
#     """Detect if an image contains NSFW content."""
#     image = Image.open(image_path).convert("RGB")
#     inputs = nsfw_processor(images=image, return_tensors="pt")
#     outputs = nsfw_model(**inputs)
#     probs = outputs.logits.softmax(dim=-1).detach().numpy()[0]
#     nsfw_score = probs[1]  # Assuming index 1 is NSFW probability
#     return nsfw_score > 0.5  # Flag as NSFW if above 50%


# NSFW Detection Function
def is_nsfw(image_path):
    """Detect if an image contains NSFW content using a specialized NSFW model."""
    predictions = predict.classify(nsfw_model, image_path)
    nsfw_score = predictions[image_path]['porn'] + predictions[image_path]['sexy']  # Summing probabilities
    if nsfw_score > 0.5:
        print(f"🚨 NSFW content detected in {image_path} (Score: {nsfw_score:.2f})")
        move_nsfw_file(image_path)
        return True
    return False

# Define dataset class
class NSFWImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root_dir, transform=transform)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, torch.tensor(label, dtype=torch.long)
    

# Define training function
def train_nsfw_model(data_dir, epochs=5, batch_size=16, lr=0.001):
    """Fine-tune the NSFW classification model."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    dataset = NSFWImageDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = AutoModelForImageClassification.from_pretrained("openai/clip-vit-base-patch32")
    model.classifier = nn.Linear(model.config.hidden_size, 2)  # Adjust for binary classification
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    # Save trained model
    model.save_pretrained("nsfw_trained_model")
    print("✅ Model training completed and saved.")

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

def is_relevant_ad(caption, category):
    """Check if the generated caption is relevant to the chosen category using similarity scoring."""
    if category.lower() not in ALLOWED_CATEGORIES:
        print("⚠️ Invalid category provided. Available categories:", list(ALLOWED_CATEGORIES.keys()))
        return False
    
    category_description = ALLOWED_CATEGORIES[category.lower()]
    similarity_score = compute_similarity(caption, category_description)
    
    print(f"🔎 Similarity Score ({category}): {similarity_score:.2f}")
    
    if similarity_score < SIMILARITY_THRESHOLD:
        return prompt_user_for_review(caption, category)
    
    return similarity_score >= SIMILARITY_THRESHOLD


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
    """Extract frames from video and analyze content."""
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    captions = []
    processed_frames = set()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % 30 == 0:  # Process one frame per second (assuming 30 FPS)
            frame_path = os.path.join("output/frames", f"{os.path.basename(video_path)}_frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            if os.path.exists(frame_path):
                if is_nsfw(frame_path):
                    if frame_path not in processed_frames:
                        print(f"🚨 NSFW content detected in {video_path} - Frame {frame_count}")
                        processed_frames.add(frame_path)
                    new_path = os.path.join("output/nsfw", os.path.basename(frame_path))
                    if not os.path.exists(new_path):
                        os.rename(frame_path, new_path)
                    else:
                        print(f"⚠️ NSFW file already exists: {new_path}, skipping rename.")
                    continue
                caption = generate_image_caption(frame_path)
                captions.append(caption)
                print(f"🖼️ Processed frame {frame_count}: {caption}")
        
        frame_count += 1
    
    cap.release()
    
    if captions:
        combined_caption = " ".join(captions)
        if is_relevant_ad(combined_caption, category):
            print(f"✅ {video_path} - Video approved under '{category}': {combined_caption}")
        else:
            print(f"❌ {video_path} - Video rejected under '{category}': {combined_caption}")

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
                continue
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
