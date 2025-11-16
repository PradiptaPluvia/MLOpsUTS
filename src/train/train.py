# simple_cnn.py
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

class SimpleShoeClassifier:
    def __init__(self):
        self.img_size = 64  # Much smaller to save memory
        self.model = None
        self.class_names = ['Boot', 'Sandal', 'Shoe']
        self.history = None
    
    def load_data_smart(self, data_dir, max_samples_per_class=1000):
        """
        Load limited data to avoid memory issues
        """
        images = []
        labels = []
        
        print("Loading data (this may take a minute)...")
        
        for class_idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(data_dir, class_name)
            if not os.path.exists(class_path):
                # Try alternative paths
                for path in ['Shoe vs Sandal vs Boot Dataset', 'archive (4)']:
                    class_path = os.path.join(path, class_name)
                    if os.path.exists(class_path):
                        break
            
            if os.path.exists(class_path):
                count = 0
                for img_name in os.listdir(class_path)[:max_samples_per_class]:
                    if count >= max_samples_per_class:
                        break
                    
                    img_path = os.path.join(class_path, img_name)
                    try:
                        # Load and resize image (smaller for memory efficiency)
                        img = cv2.imread(img_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (self.img_size, self.img_size))
                        
                        images.append(img)
                        labels.append(class_idx)
                        count += 1
                        
                    except Exception as e:
                        continue
                
                print(f"Loaded {count} {class_name} images")
        
        return np.array(images), np.array(labels)
    
    def create_simple_model(self):
        """Create a very simple CNN model"""
        model = models.Sequential([
            # Tiny conv layers
            layers.Conv2D(16, (3, 3), activation='relu', 
                         input_shape=(self.img_size, self.img_size, 3)),
            layers.MaxPooling2D(2, 2),
            
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            
            # Flatten and output
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),  # Small dropout
            layers.Dense(3, activation='softmax')
        ])
        
        # Simple optimizer with lower learning rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_lightweight(self, data_dir, epochs=5, batch_size=32):
        """Train with minimal memory usage"""
        print("Loading data...")
        X, y = self.load_data_smart(data_dir, max_samples_per_class=800)  # Limit samples
        
        if len(X) == 0:
            print("No data found! Check dataset structure.")
            return
        
        # Normalize
        X = X.astype('float32') / 255.0
        
        # Split into train/val/test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Create simple model
        self.model = self.create_simple_model()
        self.model.summary()
        
        print("Training model...")
        # Train with smaller batch size
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            verbose=1
        )
        
        # Comprehensive evaluation
        self.evaluate_model(X_test, y_test, X_val, y_val)
        
        # Save model
        self.model.save('simple_shoe_model.h5')
        print("Model saved as 'simple_shoe_model.h5'")
        
        return X_test, y_test
    
    def evaluate_model(self, X_test, y_test, X_val=None, y_val=None):
        """Comprehensive model evaluation"""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # Test set evaluation
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"📊 Test Set Accuracy: {test_accuracy:.4f}")
        print(f"📊 Test Set Loss: {test_loss:.4f}")
        
        # Validation set evaluation (if provided)
        if X_val is not None and y_val is not None:
            val_loss, val_accuracy = self.model.evaluate(X_val, y_val, verbose=0)
            print(f"📊 Validation Set Accuracy: {val_accuracy:.4f}")
            print(f"📊 Validation Set Loss: {val_loss:.4f}")
        
        # Predictions for detailed metrics
        y_pred = self.model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Classification Report
        print("\n📈 Classification Report:")
        print(classification_report(y_test, y_pred_classes, 
                                  target_names=self.class_names))
        
        # Confusion Matrix
        print("📊 Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred_classes)
        print(cm)
        
        # Per-class accuracy
        print("\n🎯 Per-class Accuracy:")
        for i, class_name in enumerate(self.class_names):
            class_mask = y_test == i
            class_accuracy = np.mean(y_pred_classes[class_mask] == y_test[class_mask])
            print(f"  {class_name}: {class_accuracy:.4f}")
        
        # Plot training history
        self.plot_training_history()
        
        # Plot confusion matrix
        self.plot_confusion_matrix(cm)
        
        # Sample predictions
        self.show_sample_predictions(X_test, y_test, y_pred_classes)
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
            
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=100, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=100, bbox_inches='tight')
        plt.show()
    
    def show_sample_predictions(self, X_test, y_test, y_pred, num_samples=5):
        """Show sample predictions"""
        print(f"\n🔍 Sample Predictions (showing {num_samples} samples):")
        
        # Get random samples
        indices = np.random.choice(len(X_test), num_samples, replace=False)
        
        for i, idx in enumerate(indices):
            true_class = self.class_names[y_test[idx]]
            pred_class = self.class_names[y_pred[idx]]
            confidence = np.max(self.model.predict(X_test[idx:idx+1], verbose=0))
            
            status = "✅ CORRECT" if true_class == pred_class else "❌ WRONG"
            print(f"Sample {i+1}: True={true_class}, Pred={pred_class}, "
                  f"Confidence={confidence:.4f} {status}")
    
    def load_and_evaluate_existing(self, data_dir):
        """Load existing model and evaluate"""
        try:
            self.model = tf.keras.models.load_model('simple_shoe_model.h5')
            print("✅ Model loaded successfully!")
        except:
            print("❌ No existing model found. Please train first.")
            return
        
        print("Loading test data...")
        X, y = self.load_data_smart(data_dir, max_samples_per_class=500)
        
        if len(X) == 0:
            print("No data found!")
            return
        
        X = X.astype('float32') / 255.0
        
        # Use a subset for evaluation to save memory
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"Evaluating on {len(X_test)} samples...")
        self.evaluate_model(X_test, y_test)

def main():
    classifier = SimpleShoeClassifier()
    
    # Find dataset
    data_dir = None
    for path in ['Shoe vs Sandal vs Boot Dataset', 'archive (4)']:
        if os.path.exists(path):
            data_dir = path
            break
    
    if data_dir is None:
        print("Dataset not found! Please run setup first.")
        print("Make sure 'archive (4).zip' is extracted.")
        return
    
    print("Choose an option:")
    print("1. Train new model and evaluate")
    print("2. Evaluate existing model")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        # Train with minimal resources
        classifier.train_lightweight(data_dir, epochs=8, batch_size=32)
    elif choice == "2":
        # Evaluate existing model
        classifier.load_and_evaluate_existing(data_dir)
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()