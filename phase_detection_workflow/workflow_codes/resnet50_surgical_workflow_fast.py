#!/usr/bin/env python3
"""
ResNet50 Frame-Level Action Recognition (FAST VERSION)
- Uses pre-extracted frames instead of reading from videos
- 10-50x faster than video-based loading
- First run extract_frames_from_videos.py to extract frames
"""
import os
_script_dir = os.path.dirname(os.path.abspath(__file__))
# Avoid "Disk quota exceeded" (matplotlib font cache and torch hub downloads to home)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_script_dir, ".cache", "matplotlib"))
os.environ.setdefault("TORCH_HOME", os.path.join(_script_dir, ".cache", "torch"))

import sys
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from datetime import datetime
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset paths (using EXTRACTED FRAMES) — relative to this script
TRAIN_DIR = os.path.join(_script_dir, "m2cai16_frames", "train")
TEST_DIR = os.path.join(_script_dir, "m2cai16_frames", "test")
ANNOTATION_TRAIN_DIR = os.path.join(_script_dir, "..", "train_dataset")
ANNOTATION_TEST_DIR = os.path.join(_script_dir, "..", "test_dataset")

# Output directory
OUTPUT_DIR = "./m2cai16_workflow_output_fast"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Timestamped logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(OUTPUT_DIR, f"training_log_{timestamp}.txt")

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

logger = Logger(log_file)
sys.stdout = logger
sys.stderr = logger

print(f"\n{'='*80}")
print(f"M2CAI16 Surgical Workflow Phase Recognition (FAST VERSION)")
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*80}\n", flush=True)

# Training configuration
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_WORKERS = 6  # Match typical HPC suggestion to avoid DataLoader slowness

# Model configuration
MODEL_NAME = "resnet50"
PRETRAINED = True
NUM_CLASSES = 8

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Configuration:")
print(f"  Train frames: {TRAIN_DIR}")
print(f"  Test frames: {TEST_DIR}")
print(f"  Output directory: {OUTPUT_DIR}")
print(f"  Model: {MODEL_NAME}")
print(f"  Pretrained: {PRETRAINED}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Num workers: {NUM_WORKERS}")
print(f"  Device: {DEVICE}\n", flush=True)

# Phase label mapping
PHASE_LABELS = {
    'TrocarPlacement': 0,
    'Preparation': 1,
    'CalotTriangleDissection': 2,
    'ClippingCutting': 3,
    'GallbladderDissection': 4,
    'GallbladderPackaging': 5,
    'CleaningCoagulation': 6,
    'GallbladderRetraction': 7,
}

LABEL_TO_PHASE = {v: k for k, v in PHASE_LABELS.items()}

print(f"Phase Labels (8 classes):")
for phase, idx in PHASE_LABELS.items():
    print(f"  {idx}: {phase}")
print(flush=True)

# ============================================================================
# DATASET (USING PRE-EXTRACTED FRAMES)
# ============================================================================

class M2CAI16FrameDataset(Dataset):
    """Dataset using pre-extracted frames (MUCH FASTER)."""
    
    def __init__(self, frames_dir, annotation_dir, transform=None, is_test=False):
        """
        Args:
            frames_dir: Directory containing extracted frames (e.g., ./m2cai16_frames/train)
            annotation_dir: Directory containing annotation files
            transform: Image transformations
            is_test: Whether this is test set
        """
        self.frames_dir = frames_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.is_test = is_test
        
        self.samples = []
        self.load_annotations()
    
    def load_annotations(self):
        """Load frame annotations and match with extracted frames."""
        print(f"\nLoading annotations from: {self.annotation_dir}")
        print(f"Looking for frames in: {self.frames_dir}")
        
        # Only use main phase-label files (workflow_video_NN.txt), not _timestamp or _pred
        prefix = "test_workflow_video_" if self.is_test else "workflow_video_"
        pattern = re.compile(r"^" + re.escape(prefix) + r"\d+\.txt$")
        all_txt = [f for f in os.listdir(self.annotation_dir) if f.endswith(".txt")]
        annotation_files = sorted([f for f in all_txt if pattern.match(f)])
        print(f"Found {len(annotation_files)} phase annotation files (excluding _timestamp, _pred)")
        
        missing_frames = 0
        
        for annotation_file in annotation_files:
            video_id = annotation_file.replace('.txt', '')
            annotation_path = os.path.join(self.annotation_dir, annotation_file)
            video_frame_dir = os.path.join(self.frames_dir, video_id)
            
            # Check if frame directory exists
            if not os.path.exists(video_frame_dir):
                print(f"Warning: Frame directory not found: {video_frame_dir}")
                print(f"  Have you run extract_frames_from_videos.py?")
                continue
            
            # Load annotations
            with open(annotation_path, 'r') as f:
                lines = f.readlines()[1:]  # Skip header
            
            # Parse annotations and match with extracted frames
            for line in lines:
                parts = line.strip().split('\t')
                if len(parts) != 2:
                    continue
                
                frame_idx = int(parts[0])
                phase_name = parts[1]
                
                # Construct frame path (matches extract_frames_from_videos.py naming)
                frame_path = os.path.join(video_frame_dir, f"frame_{frame_idx:06d}.jpg")
                
                # Check if frame exists
                if not os.path.exists(frame_path):
                    missing_frames += 1
                    continue
                
                if phase_name in PHASE_LABELS:
                    phase_label = PHASE_LABELS[phase_name]
                    self.samples.append({
                        'frame_path': frame_path,
                        'video_id': video_id,
                        'frame_idx': frame_idx,
                        'phase_label': phase_label,
                        'phase_name': phase_name
                    })
        
        print(f"Loaded {len(self.samples)} frame samples")
        if missing_frames > 0:
            print(f"Warning: {missing_frames} frames mentioned in annotations but not found on disk")
            print(f"  This is normal if frame sampling was used during extraction")
        
        # Print class distribution
        phase_counts = {}
        for sample in self.samples:
            phase_name = sample['phase_name']
            phase_counts[phase_name] = phase_counts.get(phase_name, 0) + 1
        
        print("Class distribution:")
        for phase, count in sorted(phase_counts.items(), key=lambda x: PHASE_LABELS[x[0]]):
            print(f"  {PHASE_LABELS[phase]}: {phase:30s} - {count:6d} samples ({count/len(self.samples)*100:.1f}%)")
        print(flush=True)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load frame from disk (MUCH FASTER than video extraction)
        frame = Image.open(sample['frame_path']).convert('RGB')
        
        # Apply transforms
        if self.transform:
            frame = self.transform(frame)
        
        return frame, sample['phase_label']

# ============================================================================
# DATA TRANSFORMS
# ============================================================================

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================================================
# CREATE DATASETS
# ============================================================================

print(f"\n{'='*80}")
print("Creating Datasets (from pre-extracted frames)")
print(f"{'='*80}\n")

train_dataset = M2CAI16FrameDataset(
    TRAIN_DIR,
    ANNOTATION_TRAIN_DIR,
    transform=train_transform,
    is_test=False
)

test_dataset = M2CAI16FrameDataset(
    TEST_DIR,
    ANNOTATION_TEST_DIR,
    transform=test_transform,
    is_test=True
)

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

print(f"\nDataLoaders created:")
print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
print(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches\n", flush=True)

# ============================================================================
# MODEL
# ============================================================================

print(f"\n{'='*80}")
print("Creating Model")
print(f"{'='*80}\n")

model = models.resnet50(pretrained=PRETRAINED)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, NUM_CLASSES)
model = model.to(DEVICE)

print(f"Model: ResNet50")
print(f"Pretrained: {PRETRAINED}")
print(f"Input features: {num_features}")
print(f"Output classes: {NUM_CLASSES}\n", flush=True)

# ============================================================================
# TRAINING SETUP
# ============================================================================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=5,
)

print(f"Training Setup:")
print(f"  Loss: CrossEntropyLoss")
print(f"  Optimizer: Adam (lr={LEARNING_RATE})")
print(f"  Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)\n", flush=True)

# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, all_preds, all_labels

# ============================================================================
# TRAINING
# ============================================================================

print(f"\n{'='*80}")
print("Starting Training")
print(f"{'='*80}\n")

best_acc = 0.0
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    print("-" * 40)
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    
    val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion, DEVICE)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    print(f"\nResults:")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    scheduler.step(val_acc)
    
    if val_acc > best_acc:
        best_acc = val_acc
        checkpoint_path = os.path.join(OUTPUT_DIR, "best_model.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
        }, checkpoint_path)
        print(f"  >>> New best model saved! (Acc: {val_acc:.4f})")
    
    print(flush=True)

print(f"\n{'='*80}")
print("Training Complete")
print(f"Best Validation Accuracy: {best_acc:.4f}")
print(f"{'='*80}\n", flush=True)

# Save training history
history_path = os.path.join(OUTPUT_DIR, "training_history.json")
with open(history_path, 'w') as f:
    json.dump(history, f, indent=2)

# Final evaluation (same as original script)
print(f"\n{'='*80}")
print("Final Evaluation on Test Set")
print(f"{'='*80}\n")

checkpoint = torch.load(os.path.join(OUTPUT_DIR, "best_model.pth"))
model.load_state_dict(checkpoint['model_state_dict'])

test_loss, test_acc, all_preds, all_labels = evaluate(model, test_loader, criterion, DEVICE)

print(f"\nTest Results:")
print(f"  Test Loss: {test_loss:.4f}")
print(f"  Test Accuracy: {test_acc:.4f}\n")

precision, recall, f1, support = precision_recall_fscore_support(
    all_labels, all_preds, average=None, zero_division=0
)

print("Per-Class Metrics:")
print(f"{'Class':<30s} {'Precision':<10s} {'Recall':<10s} {'F1-Score':<10s} {'Support':<10s}")
print("-" * 70)
for i in range(NUM_CLASSES):
    phase_name = LABEL_TO_PHASE[i]
    print(f"{phase_name:<30s} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f} {support[i]:<10.0f}")

precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
    all_labels, all_preds, average='weighted', zero_division=0
)

print("-" * 70)
print(f"{'Weighted Average':<30s} {precision_avg:<10.4f} {recall_avg:<10.4f} {f1_avg:<10.4f}")
print(flush=True)

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=[LABEL_TO_PHASE[i] for i in range(NUM_CLASSES)],
    yticklabels=[LABEL_TO_PHASE[i] for i in range(NUM_CLASSES)]
)
plt.title('Confusion Matrix - Surgical Workflow Phase Recognition')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=300, bbox_inches='tight')

# Training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(history['train_loss'], label='Train Loss')
ax1.plot(history['val_loss'], label='Val Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True)

ax2.plot(history['train_acc'], label='Train Accuracy')
ax2.plot(history['val_acc'], label='Val Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Training and Validation Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=300, bbox_inches='tight')

# Save results
results = {
    'best_val_acc': best_acc,
    'test_acc': test_acc,
    'test_loss': test_loss,
    'per_class_metrics': {
        LABEL_TO_PHASE[i]: {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }
        for i in range(NUM_CLASSES)
    },
    'weighted_avg': {
        'precision': float(precision_avg),
        'recall': float(recall_avg),
        'f1': float(f1_avg)
    },
    'config': {
        'model': MODEL_NAME,
        'batch_size': BATCH_SIZE,
        'epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'num_workers': NUM_WORKERS
    }
}

with open(os.path.join(OUTPUT_DIR, "results.json"), 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*80}")
print("All Complete!")
print(f"Ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"{'='*80}\n", flush=True)
