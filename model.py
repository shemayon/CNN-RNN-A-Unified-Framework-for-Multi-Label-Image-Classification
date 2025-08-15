import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class CNNRNNMultiLabel(nn.Module):
    """
    CNN-RNN: A Unified Framework for Multi-Label Image Classification
    Based on the paper: https://arxiv.org/pdf/1604.04573
    """
    def __init__(self, num_classes, embed_size=512, hidden_size=512, num_layers=1, dropout=0.5):
        super(CNNRNNMultiLabel, self).__init__()
        
        # CNN Encoder (ResNet-152)
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]  # Remove the last fc layer
        self.cnn_encoder = nn.Sequential(*modules)
        
        # Feature projection layer
        self.feature_projection = nn.Linear(resnet.fc.in_features, embed_size)
        self.feature_norm = nn.LayerNorm(embed_size)  # Use LayerNorm instead of BatchNorm1d
        
        # Label embedding layer (add 1 for padding token)
        self.label_embedding = nn.Embedding(num_classes + 1, embed_size)
        
        # RNN for modeling label dependencies
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, 
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Output projection layers
        self.image_projection = nn.Linear(embed_size, hidden_size)
        self.label_projection = nn.Linear(hidden_size, embed_size)
        
        # Final classification layer
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize the weights of the model"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def encode_image(self, images):
        """Extract features from images using CNN"""
        features = self.cnn_encoder(images)
        features = features.view(features.size(0), -1)
        features = self.feature_projection(features)
        features = self.feature_norm(features)
        features = F.relu(features)
        features = self.dropout(features)
        return features
    
    def forward(self, images, label_sequence=None, label_lengths=None):
        """
        Forward pass of the CNN-RNN model
        
        Args:
            images: Input images [batch_size, 3, H, W]
            label_sequence: Label sequence for training [batch_size, seq_len]
            label_lengths: Length of each label sequence [batch_size]
        
        Returns:
            outputs: Classification scores [batch_size, num_classes]
            hidden_states: RNN hidden states for analysis
        """
        batch_size = images.size(0)
        
        # Encode images
        image_features = self.encode_image(images)  # [batch_size, embed_size]
        
        if self.training and label_sequence is not None and len(label_sequence) > 0:
            # Training mode: use teacher forcing with label sequences
            try:
                # Pad sequences to the same length
                max_length = max(len(seq) for seq in label_sequence)
                padded_sequences = []
                
                for seq in label_sequence:
                    if len(seq) < max_length:
                        # Pad with zeros (assuming 0 is not a valid label index)
                        padded_seq = torch.cat([seq, torch.zeros(max_length - len(seq), dtype=seq.dtype, device=seq.device)])
                    else:
                        padded_seq = seq
                    padded_sequences.append(padded_seq)
            except Exception as e:
                print(f"Error processing label sequences: {e}")
                # Fall back to inference mode
                label_sequence = None
            
            # Stack padded sequences
            padded_sequences = torch.stack(padded_sequences)  # [batch_size, max_length]
            
            # Embed labels
            label_embeddings = self.label_embedding(padded_sequences)  # [batch_size, seq_len, embed_size]
            
            # Initialize RNN hidden state with image features
            h0 = self.image_projection(image_features).unsqueeze(0).repeat(self.rnn.num_layers, 1, 1)
            c0 = torch.zeros_like(h0)
            
            # Process label sequence through RNN
            packed_embeddings = nn.utils.rnn.pack_padded_sequence(
                label_embeddings, label_lengths, batch_first=True, enforce_sorted=False
            )
            
            rnn_output, (hidden, cell) = self.rnn(packed_embeddings, (h0, c0))
            
            # Unpack the output
            rnn_output, _ = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
            
            # Use the last hidden state for classification
            last_hidden = hidden[-1]  # [batch_size, hidden_size]
            
        else:
            # Inference mode: use image features to initialize RNN
            h0 = self.image_projection(image_features).unsqueeze(0).repeat(self.rnn.num_layers, 1, 1)
            c0 = torch.zeros_like(h0)
            
            # Create dummy input for RNN (we only need the hidden state)
            dummy_input = torch.zeros(batch_size, 1, self.label_embedding.embedding_dim).to(images.device)
            
            _, (hidden, cell) = self.rnn(dummy_input, (h0, c0))
            last_hidden = hidden[-1]  # [batch_size, hidden_size]
            rnn_output = None
        
        # Final classification
        outputs = self.classifier(last_hidden)  # [batch_size, num_classes]
        
        return outputs, last_hidden
    
    def predict_labels(self, images, threshold=0.5):
        """
        Predict labels for given images
        
        Args:
            images: Input images [batch_size, 3, H, W]
            threshold: Classification threshold
        
        Returns:
            predictions: Binary predictions [batch_size, num_classes]
            scores: Raw classification scores [batch_size, num_classes]
        """
        self.eval()
        with torch.no_grad():
            scores, _ = self.forward(images)
            predictions = (torch.sigmoid(scores) > threshold).float()
        return predictions, scores


    
