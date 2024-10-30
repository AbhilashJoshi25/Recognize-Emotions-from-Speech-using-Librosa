import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import os

def extract_features(audio_path):
    """Extract audio features using librosa."""
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, duration=3, offset=0.5)
        
        # Extract features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_scaled = np.mean(chroma.T, axis=0)
        
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_scaled = np.mean(mel.T, axis=0)
        
        # Extract additional features
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        energy = np.array([sum(abs(y))/len(y)])
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Combine all features
        features = np.concatenate([
            mfcc_scaled,
            chroma_scaled,
            mel_scaled,
            zero_crossing_rate.flatten(),
            energy,
            [tempo]
        ])
        
        return features
        
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {str(e)}")
        return None

class EmotionRecognizer:
    def __init__(self):
        self.classifier = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            max_iter=500
        )
        self.emotions = ['angry', 'happy', 'sad', 'neutral']
        
    def train(self, data_path):
        """Train the emotion recognizer with audio files from the specified path."""
        features = []
        labels = []
        
        # Iterate through each emotion folder
        for emotion in self.emotions:
            emotion_path = os.path.join(data_path, emotion)
            if not os.path.exists(emotion_path):
                continue
                
            # Process each audio file in the emotion folder
            for filename in os.listdir(emotion_path):
                if filename.endswith('.wav'):
                    file_path = os.path.join(emotion_path, filename)
                    feature_vector = extract_features(file_path)
                    
                    if feature_vector is not None:
                        features.append(feature_vector)
                        labels.append(self.emotions.index(emotion))
        
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train the classifier
        self.classifier.fit(X_train, y_train)
        
        # Calculate and return accuracy
        train_accuracy = self.classifier.score(X_train, y_train)
        test_accuracy = self.classifier.score(X_test, y_test)
        
        return train_accuracy, test_accuracy
    
    def predict(self, audio_path):
        """Predict emotion for a single audio file."""
        features = extract_features(audio_path)
        if features is None:
            return None
            
        # Reshape features for prediction
        features = features.reshape(1, -1)
        
        # Get prediction and probability
        prediction = self.classifier.predict(features)[0]
        probabilities = self.classifier.predict_proba(features)[0]
        
        # Create result dictionary
        result = {
            'emotion': self.emotions[prediction],
            'confidence': probabilities[prediction],
            'probabilities': {
                emotion: prob 
                for emotion, prob in zip(self.emotions, probabilities)
            }
        }
        
        return result

# Example usage
if __name__ == "__main__":
    # Initialize the emotion recognizer
    recognizer = EmotionRecognizer()
    
    # Example paths (you would need to provide your own data)
    data_path = "path/to/emotion/dataset"
    test_audio = "path/to/test/audio.wav"
    
    # Train the model
    print("Training model...")
    train_acc, test_acc = recognizer.train(data_path)
    print(f"Training accuracy: {train_acc:.2f}")
    print(f"Testing accuracy: {test_acc:.2f}")
    
    # Predict emotion for a test file
    print("\nPredicting emotion...")
    result = recognizer.predict(test_audio)
    if result:
        print(f"Predicted emotion: {result['emotion']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print("\nProbabilities for all emotions:")
        for emotion, prob in result['probabilities'].items():
            print(f"{emotion}: {prob:.2f}")
