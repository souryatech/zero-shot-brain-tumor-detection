import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor, CLIPConfig
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from open_clip import create_model_from_pretrained, get_tokenizer


class ZeroShotObjectDetector(nn.Module):
    """
    Zero-shot object detection using CLIP-based models.
    
    Architecture:
    - Text Encoder: BiomedCLIP (PubMedBERT + ViT)
    - Image Encoder: CLIP (ViT-base)
    - Similarity computation using cosine similarity
    - Bounding box generation from similarity scores
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device
        
        # Initialize BiomedCLIP for text encoding
        print("Loading BiomedCLIP...")
        self.text_model, self.preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.text_tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        
        # Initialize CLIP for image encoding
        print("Loading CLIP ViT-base...")
        config = CLIPConfig.from_pretrained('openai/clip-vit-base-patch32')

        self.image_model = CLIPModel(config)
        self.image_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        
        # Move models to device
        self.text_model = self.text_model.to(device)
        self.image_model = self.image_model.to(device)
        
        # Freeze models (no training needed for zero-shot)
        self.text_model.eval()
        self.image_model.eval()
        
        # Projection layers to align embedding dimensions
        self.text_projection = nn.Linear(512, 512).to(device)  # BiomedCLIP to CLIP dimension
        self.image_projection = nn.Linear(768, 512).to(device)  # CLIP dimension alignment

        
        
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Encode text descriptions using BiomedCLIP.
        
        Args:
            texts: List of text descriptions
            
        Returns:
            Text embeddings of shape (batch_size, embedding_dim)
        """
        # Tokenize texts
        inputs = self.text_tokenizer(
            texts
        ).to(self.device)
        
        # Get text embeddings
        with torch.no_grad():
            outputs = self.text_model.encode_text(inputs)
            text_embeddings = outputs
            
        # Project to common dimension
        text_embeddings = self.text_projection(text_embeddings)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        
        return text_embeddings
    
    def encode_image_patches(self, image: torch.Tensor, patch_size: int = 32) -> Tuple[torch.Tensor, List[Tuple[int, int, int, int]]]:
        """
        Encode image patches using CLIP ViT-base.
        
        Args:
            image: Input image tensor of shape (3, H, W)
            patch_size: Size of patches to extract
            
        Returns:
            - Patch embeddings of shape (num_patches, embedding_dim)
            - List of patch coordinates (x_min, y_min, x_max, y_max)
        """
        # For individual image
        if image.ndim == 4:  # [B, C, H, W]
            image = image[0]  # Remove batch dimension

            
        _, H, W = image.shape
        patches = []
        coordinates = []
        
        # Extract overlapping patches with stride
        stride = patch_size // 2
        
        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                # Extract patch
                patch = image[:, y:y+patch_size, x:x+patch_size]
                patches.append(patch)
                coordinates.append((x, y, x+patch_size, y+patch_size))
        
        # Process patches through CLIP
        if patches:
            patches_tensor = torch.stack(patches)
            
            # Resize patches to CLIP input size (224x224)
            resize = transforms.Resize((224, 224))
            patches_resized = torch.stack([resize(p) for p in patches_tensor])
            
            # Normalize patches
            normalize = transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
            patches_normalized = torch.stack([normalize(p) for p in patches_resized])
            
            # Get patch embeddings
            with torch.no_grad():
                inputs = {'pixel_values': patches_normalized.to(self.device)}
                outputs = self.image_model.vision_model(**inputs)
                patch_embeddings = outputs.pooler_output
                
            # Project embeddings
            patch_embeddings = self.image_projection(patch_embeddings)
            patch_embeddings = F.normalize(patch_embeddings, p=2, dim=-1)
            
            return patch_embeddings, coordinates
        
        return torch.empty(0, 512).to(self.device), []
    
    def compute_similarity_map(self, image: torch.Tensor, text_embedding: torch.Tensor, 
                             patch_size: int = 32) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        """
        Compute similarity map between image patches and text embedding.
        
        Args:
            image: Input image tensor
            text_embedding: Text embedding vector
            patch_size: Size of patches
            
        Returns:
            - Similarity scores array
            - List of patch coordinates
        """
        # Get patch embeddings
        patch_embeddings, coordinates = self.encode_image_patches(image, patch_size)
        
        if len(patch_embeddings) == 0:
            return np.array([]), []
        
        # Compute cosine similarity
        similarities = torch.matmul(patch_embeddings, text_embedding.unsqueeze(-1)).squeeze(-1)
        
        return similarities.cpu().numpy(), coordinates
    
    def generate_bounding_boxes(self, similarity_scores: np.ndarray, 
                              coordinates: List[Tuple[int, int, int, int]], 
                              threshold: float = 0.5) -> List[Dict]:
        """
        Generate bounding boxes from similarity scores.
        
        Args:
            similarity_scores: Array of similarity scores
            coordinates: List of patch coordinates
            threshold: Similarity threshold for box generation
            
        Returns:
            List of bounding boxes with scores
        """
        boxes = []
        
        # Filter by threshold
        high_sim_indices = np.where(similarity_scores > threshold)[0]
        
        for idx in high_sim_indices:
            x_min, y_min, x_max, y_max = coordinates[idx]
            score = float(similarity_scores[idx])
            
            boxes.append({
                'box': [x_min, y_min, x_max, y_max],
                'score': score
            })
        
        return boxes
    
    def apply_nms(self, boxes: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """
        Apply Non-Maximum Suppression to remove overlapping boxes.
        
        Args:
            boxes: List of bounding boxes with scores
            iou_threshold: IoU threshold for NMS
            
        Returns:
            Filtered list of bounding boxes
        """
        if not boxes:
            return []
        
        # Convert to numpy arrays
        boxes_array = np.array([b['box'] for b in boxes])
        scores_array = np.array([b['score'] for b in boxes])
        
        # Apply NMS
        indices = self.non_max_suppression(boxes_array, scores_array, iou_threshold)
        
        return [boxes[i] for i in indices]
    
    def non_max_suppression(self, boxes: np.ndarray, scores: np.ndarray, 
                          iou_threshold: float) -> List[int]:
        """
        Perform Non-Maximum Suppression algorithm.
        """
        if len(boxes) == 0:
            return []
        
        # Sort boxes by score (descending)
        sorted_indices = np.argsort(scores)[::-1]
        
        keep = []
        while len(sorted_indices) > 0:
            # Take the box with highest score
            current = sorted_indices[0]
            keep.append(current)
            
            if len(sorted_indices) == 1:
                break
            
            # Compute IoU with remaining boxes
            current_box = boxes[current]
            remaining_boxes = boxes[sorted_indices[1:]]
            ious = self.compute_iou(current_box, remaining_boxes)
            
            # Keep boxes with IoU less than threshold
            indices_to_keep = np.where(ious < iou_threshold)[0] + 1
            sorted_indices = sorted_indices[indices_to_keep]
        
        return keep
    
    def compute_iou(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        Compute IoU between one box and multiple boxes.
        """
        # Compute intersection
        x_min = np.maximum(box[0], boxes[:, 0])
        y_min = np.maximum(box[1], boxes[:, 1])
        x_max = np.minimum(box[2], boxes[:, 2])
        y_max = np.minimum(box[3], boxes[:, 3])
        
        intersection = np.maximum(0, x_max - x_min) * np.maximum(0, y_max - y_min)
        
        # Compute union
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = box_area + boxes_area - intersection
        
        # Compute IoU
        iou = intersection / (union + 1e-6)
        
        return iou
    
    def forward(self, image: torch.Tensor, text: str, 
                similarity_threshold: float = 0.5,
                nms_threshold: float = 0.5) -> List[Dict]:
        """
        Perform zero-shot object detection.
        
        Args:
            image: Input image tensor
            text: Text description of object to detect
            similarity_threshold: Threshold for similarity scores
            nms_threshold: Threshold for NMS
            
        Returns:
            List of detected bounding boxes with scores
        """
        # Encode text
        
        text_embedding = self.encode_text([text])[0]
        
        # Compute similarity map
        similarity_scores, coordinates = self.compute_similarity_map(
            image, text_embedding
        )
        
        # Generate bounding boxes
        boxes = self.generate_bounding_boxes(
            similarity_scores, coordinates, similarity_threshold
        )
        
        # Apply NMS
        filtered_boxes = self.apply_nms(boxes, nms_threshold)
        
        return filtered_boxes


def create_text_prompt(class_name: str, description: str = None, 
                      prompt_strategy: str = 'template') -> str:
    """
    Create text prompt for the encoder based on different strategies.
    
    Args:
        class_name: Object class name
        description: MRI scan description
        prompt_strategy: How to combine class and description
            - 'description_only': Use only the description
            - 'class_only': Use only the class name
            - 'combined': Combine both
            - 'template': Use a template format
            
    Returns:
        Text prompt string
    """
    if prompt_strategy == 'description_only' and description:
        return description
    elif prompt_strategy == 'class_only':
        return class_name
    elif prompt_strategy == 'combined' and description:
        return f"{class_name}: {description}"
    elif prompt_strategy == 'template' and description:
        return f"MRI scan showing {class_name}. {description}"
    else:
        # Fallback to class name if description is not available
        return class_name


class ZeroShotDataset(Dataset):
    """
    Dataset class for zero-shot object detection.
    
    Expects CSV with columns: img_id, class, description, x_min, y_min, x_max, y_max
    """
    
    def __init__(self, csv_path: str, img_dir: str, transform=None, use_description=True):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.use_description = use_description
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load image
        img_path = f"{self.img_dir}/{row['image_id']}"
        image = Image.open(img_path).convert('RGB')
        
        # Get ground truth
        gt_box = [row['x_min'], row['y_min'], row['x_max'], row['y_max']]
        class_name = row['class']
        
        # Get description - use it if available, otherwise fall back to class name
        if self.use_description and 'description' in row and pd.notna(row['description']):
            text_prompt = row['description']
        else:
            text_prompt = class_name
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        return {
            'image': image,
            'class_name': class_name,
            'text_prompt': text_prompt,
            'gt_box': torch.tensor(gt_box, dtype=torch.float32),
            'img_id': row['image_id']
        }


class SymmetricalCrossEntropyLoss(nn.Module):
    """
    Symmetrical Cross-Entropy Loss for zero-shot learning.
    
    This loss encourages alignment between image and text embeddings.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute symmetrical cross-entropy loss.
        
        Args:
            image_embeddings: Image embeddings (batch_size, embedding_dim)
            text_embeddings: Text embeddings (batch_size, embedding_dim)
            
        Returns:
            Loss value
        """
        batch_size = image_embeddings.shape[0]
        
        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        
        # Compute similarity matrix
        logits = torch.matmul(image_embeddings, text_embeddings.t()) / self.temperature
        
        # Create labels (diagonal elements are positive pairs)
        labels = torch.arange(batch_size).to(logits.device)
        
        # Compute cross-entropy in both directions
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        
        # Symmetrical loss
        loss = (loss_i2t + loss_t2i) / 2
        
        return loss


def evaluate_detector(detector: ZeroShotObjectDetector, dataloader: DataLoader, 
                     iou_threshold: float = 0.5) -> Dict[str, float]:
    """
    Evaluate detector using Average Precision @ IoU = 0.5.
    
    Args:
        detector: Zero-shot detector model
        dataloader: DataLoader for evaluation
        iou_threshold: IoU threshold for matching
        
    Returns:
        Dictionary with evaluation metrics
    """
    detector.eval()
    
    all_predictions = []
    all_ground_truths = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['image'].to(detector.device)
            text_prompts = batch['text_prompt']  # Use text_prompt instead of class_name
            gt_boxes = batch['gt_box']
            
            # Process each image
            for i in range(len(images)):
                # Get predictions using the description
                predictions = detector(images[i], text_prompts[i])
                
                # Store predictions and ground truths
                if predictions:
                    pred_boxes = np.array([p['box'] for p in predictions])
                    pred_scores = np.array([p['score'] for p in predictions])
                    all_predictions.append({
                        'boxes': pred_boxes,
                        'scores': pred_scores
                    })
                else:
                    all_predictions.append({
                        'boxes': np.array([]),
                        'scores': np.array([])
                    })
                
                all_ground_truths.append(gt_boxes[i].numpy())
    
    # Compute Average Precision
    ap = compute_average_precision(all_predictions, all_ground_truths, iou_threshold)
    
    return {'AP@IoU=0.5': ap}


def compute_average_precision(predictions: List[Dict], ground_truths: List[np.ndarray], 
                            iou_threshold: float) -> float:
    """
    Compute Average Precision for object detection.
    """
    # Implementation of AP calculation
    # This is a simplified version - you may want to use a library like COCO API
    
    true_positives = 0
    false_positives = 0
    total_gt = len(ground_truths)
    
    for pred, gt in zip(predictions, ground_truths):
        if len(pred['boxes']) == 0:
            continue
            
        # Find best matching prediction
        best_iou = 0
        for pred_box in pred['boxes']:
            iou = compute_single_iou(pred_box, gt)
            best_iou = max(best_iou, iou)
        
        if best_iou >= iou_threshold:
            true_positives += 1
        else:
            false_positives += 1
    
    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (total_gt + 1e-6)
    
    # Simplified AP (you may want to compute full precision-recall curve)
    ap = precision * recall
    
    return ap


def compute_single_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute IoU between two boxes.
    """
    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3])
    
    intersection = max(0, x_max - x_min) * max(0, y_max - y_min)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / (union + 1e-6)


def visualize_detections(image: torch.Tensor, boxes: List[Dict], class_name: str):
    """
    Visualize detected bounding boxes on image.
    """
    # Convert tensor to numpy
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
    
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    # Draw bounding boxes
    for box_dict in boxes:
        box = box_dict['box']
        score = box_dict['score']
        
        rect = patches.Rectangle(
            (box[0], box[1]), 
            box[2] - box[0], 
            box[3] - box[1],
            linewidth=2, 
            edgecolor='red', 
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add score text
        ax.text(
            box[0], box[1] - 5,
            f'{class_name}: {score:.2f}',
            color='red',
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5)
        )
    
    ax.set_title(f'Zero-Shot Detection: {class_name}')
    ax.axis('off')
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = ZeroShotObjectDetector()
    
    # Example: Load and process a single image with custom description
    image_path = "example.jpg"
    image = Image.open(image_path).convert('RGB')
    image_tensor = transforms.ToTensor()(image)
    
    # Detect objects using a detailed description
    description = "T2-weighted axial MRI showing hyperintense lesion in the left temporal lobe"
    detections = detector(image_tensor, description)
    
    # Visualize results
    visualize_detections(image_tensor, detections, "lesion")
    
    # For training/evaluation with dataset
    # CSV should have columns: img_id, class, description, x_min, y_min, x_max, y_max
    # dataset = ZeroShotDataset('train.csv', 'images/', use_description=True)
    # dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    # metrics = evaluate_detector(detector, dataloader)
    # print(f"Evaluation metrics: {metrics}")
    
    # Example of processing with both class name and description
    # You can also create custom prompts combining both:
    # combined_prompt = f"{class_name}: {description}"
    # detections = detector(image_tensor, combined_prompt)