import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from nltk.tokenize import word_tokenize
from collections import Counter
import nltk
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# 1. Data processing and loading
class Flickr8kDataset(Dataset):
    def __init__(self, image_dir, captions_df, image_ids, transform=None, max_caption_length=30):
        self.image_dir = image_dir
        self.df = captions_df[captions_df['image'].isin(image_ids)].reset_index(drop=True)
        self.transform = transform
        self.max_caption_length = max_caption_length

        # create vocabulary
        self.captions = self.df['caption'].tolist()
        self.vocab = self._build_vocab()
        self.word2idx = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}
        self.word2idx.update({word: idx+4 for idx, word in enumerate(self.vocab)})
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def _build_vocab(self, min_freq=2):
        counter = Counter()
        for caption in self.captions:
            tokens = word_tokenize(caption)
            counter.update(tokens)
        return [word for word, cnt in counter.items() if cnt >= min_freq]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row['image'])
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        caption = row['caption']
        tokens = ['<start>'] + word_tokenize(caption) + ['<end>']
        token_ids = [self.word2idx.get(token, self.word2idx['<unk>']) for token in tokens]

        if len(token_ids) < self.max_caption_length:
            token_ids += [self.word2idx['<pad>']] * (self.max_caption_length - len(token_ids))
        else:
            token_ids = token_ids[:self.max_caption_length]

        return image, torch.tensor(token_ids)

def load_and_filter_captions(token_file):
    """Load and filter text descriptions, retaining only the # 2 description for each image"""
    captions = []
    with open(token_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                img_id, caption_num = parts[0].split('#')
                if caption_num == '2':
                    captions.append({'image': img_id, 'caption': parts[1].lower()})
    return pd.DataFrame(captions)

def load_split_image_ids(split_file):
    """List of Image IDs for Loading Partition Files"""
    with open(split_file, 'r') as f:
        return [line.strip() for line in f]

def prepare_datasets():
    """Prepare training set, validation set, and test set"""
    # Set file path
    IMAGE_DIR = '/root/autodl-tmp/Flickr8k_Dataset/Flicker8k_Dataset'
    TOKEN_FILE = '/root/autodl-tmp/Flickr8k_text/Flickr8k.token.txt'
    TRAIN_SPLIT_FILE = '/root/autodl-tmp/Flickr8k_text/Flickr_8k.trainImages.txt'
    DEV_SPLIT_FILE = '/root/autodl-tmp/Flickr8k_text/Flickr_8k.devImages.txt'
    TEST_SPLIT_FILE = '/root/autodl-tmp/Flickr8k_text/Flickr_8k.testImages.txt'

    # Load data
    captions_df = load_and_filter_captions(TOKEN_FILE)
    train_ids = load_split_image_ids(TRAIN_SPLIT_FILE)
    dev_ids = load_split_image_ids(DEV_SPLIT_FILE)
    test_ids = load_split_image_ids(TEST_SPLIT_FILE)

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset
    train_dataset = Flickr8kDataset(IMAGE_DIR, captions_df, train_ids, transform)
    dev_dataset = Flickr8kDataset(IMAGE_DIR, captions_df, dev_ids, transform)
    test_dataset = Flickr8kDataset(IMAGE_DIR, captions_df, test_ids, transform)

    return train_dataset, dev_dataset, test_dataset

def create_data_loaders(train_dataset, dev_dataset, test_dataset, batch_size=32):
    """创建DataLoader"""
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    dev_loader = DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, dev_loader, test_loader

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        # encoder_out: (batch_size, num_pixels, encoder_dim)
        # decoder_hidden: (1, batch_size, decoder_dim)

        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden.squeeze(0))  # (batch_size, attention_dim)

        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return context, alpha

#  Model Definition
class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_size=512, hidden_size=512, attention_dim=256):
        super().__init__()
        # Image Encoder
        self.encoder = models.efficientnet_b0(pretrained=True)
        self.encoder.classifier = nn.Sequential(
            nn.Linear(self.encoder.classifier[1].in_features, embed_size),
            nn.ReLU(),
            nn.Dropout(0.5))

        # Text decoder component
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(embed_size, hidden_size, attention_dim)
        self.lstm = nn.LSTM(embed_size + embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, images, captions, lengths):
        # Encode image features
        features = self.encoder(images)  # (batch_size, embed_size)
        features = features.unsqueeze(1)  # (batch_size, 1, embed_size)

        # Embedded Text
        embeddings = self.embedding(captions)  # (batch_size, seq_len, embed_size)

        # Initialize LSTM state
        batch_size = images.size(0)
        h = torch.zeros(1, batch_size, self.lstm.hidden_size).to(images.device)
        c = torch.zeros(1, batch_size, self.lstm.hidden_size).to(images.device)

        outputs = torch.zeros(batch_size, max(lengths), self.fc.out_features).to(images.device)

        for t in range(max(lengths)):
            # #Calculate attention context
            context, _ = self.attention(features, h)  # context: (batch_size, embed_size)

            # Prepare LSTM input
            lstm_input = torch.cat([
                embeddings[:, t, :],  # (batch_size, embed_size)
                context  # (batch_size, embed_size)
            ], dim=1).unsqueeze(1)  # (batch_size, 1, 2*embed_size)

            # LSTM forward propagation
            lstm_out, (h, c) = self.lstm(lstm_input, (h, c))

            # #Predict the next word
            output = self.fc(self.dropout(lstm_out.squeeze(1)))
            outputs[:, t, :] = output

        return outputs

    def generate_caption(self, image, word2idx, idx2word, max_length=30, device='cuda'):
        """Generate image description"""
        self.eval()
        with torch.no_grad():
            # Ensure that the image shape is correct [1, 3, 224, 224]
            if image.dim() == 4:  # 如果已经是 [1, 3, 224, 224]
                img_tensor = image.to(device)
            else:  # If [3, 224, 224] 或 [224, 224, 3]
                img_tensor = image.unsqueeze(0).to(device)

            # Encode image features
            features = self.encoder(img_tensor)  # (1, embed_size)
            features = features.unsqueeze(1)  # (1, 1, embed_size)

            # Initialize LSTM state
            h = torch.zeros(1, 1, self.lstm.hidden_size).to(device)
            c = torch.zeros(1, 1, self.lstm.hidden_size).to(device)

            # Start generating description
            sampled_ids = [word2idx['<start>']]
            for _ in range(max_length):
                inputs = torch.LongTensor([sampled_ids[-1]]).to(device)
                embeddings = self.embedding(inputs).unsqueeze(1)  # (1, 1, embed_size)

                # Calculate attention context
                context, _ = self.attention(features, h)  # context: (1, embed_size)

                # LSTM forward propagation
                lstm_input = torch.cat([embeddings, context.unsqueeze(1)], dim=2)  # (1, 1, 2*embed_size)
                lstm_out, (h, c) = self.lstm(lstm_input, (h, c))

                # Predict the next word
                output = self.fc(self.dropout(lstm_out.squeeze(1)))
                _, predicted = output.max(1)
                sampled_ids.append(predicted.item())

                if predicted.item() == word2idx['<end>']:
                    break

            # Convert token to word
            caption = [idx2word[idx] for idx in sampled_ids
                      if idx not in [word2idx['<start>'], word2idx['<end>'], word2idx['<pad>']]]
            return caption

# 3. Training and evaluation functions
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, captions in tqdm(dataloader):
        images = images.to(device)
        captions = captions.to(device)

        optimizer.zero_grad()
        outputs = model(images, captions[:, :-1], [captions.size(1)-1]*images.size(0))
        loss = criterion(outputs.view(-1, outputs.size(-1)), captions[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, word2idx, idx2word, device):
    model.eval()
    references = []
    hypotheses = []
    smoothie = SmoothingFunction().method4

    with torch.no_grad():
        for images, captions in tqdm(dataloader):
            images = images.to(device)

            # Generate description
            for i in range(images.size(0)):
                img = images[i]  
                caption = model.generate_caption(img, word2idx, idx2word, device=device)
                hypotheses.append(caption)

                # Get reference description
                ref = [idx2word[idx.item()] for idx in captions[i]
                      if idx.item() not in [word2idx['<start>'], word2idx['<end>'], word2idx['<pad>']]]
                references.append([ref])

    # Calculate BLEU score
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0), smoothing_function=smoothie)
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

    return bleu1, bleu2, bleu3, bleu4

# Main training process
if __name__ == '__main__':
    # Prepare data
    train_dataset, dev_dataset, test_dataset = prepare_datasets()
    train_loader, dev_loader, test_loader = create_data_loaders(train_dataset, dev_dataset, test_dataset)

    # Initial model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = len(train_dataset.word2idx)
    model = ImageCaptioningModel(vocab_size).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.word2idx['<pad>'])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training cycle
    num_epochs = 100
    best_bleu4 = 0.0

    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}')

        # 验证集评估
        bleu1, bleu2, bleu3, bleu4 = evaluate(model, dev_loader,
                                             train_dataset.word2idx, train_dataset.idx2word, device)
        print(f'Validation BLEU Scores:')
        print(f'BLEU-1: {bleu1:.4f}, BLEU-2: {bleu2:.4f}, BLEU-3: {bleu3:.4f}, BLEU-4: {bleu4:.4f}')

        # Save the best model
        if bleu4 > best_bleu4:
            best_bleu4 = bleu4
            torch.save(model.state_dict(), 'best_model.pth')
            print('New best model saved!')

    # Test set evaluation
    model.load_state_dict(torch.load('best_model.pth'))
    bleu1, bleu2, bleu3, bleu4 = evaluate(model, test_loader,
                                         train_dataset.word2idx, train_dataset.idx2word, device)
    print(f'\\nTest BLEU Scores:')
    print(f'BLEU-1: {bleu1:.4f}, BLEU-2: {bleu2:.4f}, BLEU-3: {bleu3:.4f}, BLEU-4: {bleu4:.4f}')

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_size = len(train_dataset.word2idx)
model = ImageCaptioningModel(vocab_size).to(device)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()    
