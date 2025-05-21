from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
def generate_caption_for_image(image_path, model, transform, word2idx, idx2word, device='cuda'):
    # Load and preprocess images
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # Generate description
    with torch.no_grad():
        features = model.encoder(image)
        features = features.unsqueeze(1)

        h = torch.zeros(1, 1, model.lstm.hidden_size).to(device)
        c = torch.zeros(1, 1, model.lstm.hidden_size).to(device)

        sampled_ids = [word2idx['<start>']]
        for _ in range(30):  # Maximum generated length
            inputs = torch.LongTensor([sampled_ids[-1]]).to(device)
            embeddings = model.embedding(inputs).unsqueeze(1)

            context, _ = model.attention(features, h)

            lstm_input = torch.cat([embeddings, context.unsqueeze(1)], dim=2)
            lstm_out, (h, c) = model.lstm(lstm_input, (h, c))

            output = model.fc(model.dropout(lstm_out.squeeze(1)))
            _, predicted = output.max(1)
            sampled_ids.append(predicted.item())

            if predicted.item() == word2idx['<end>']:
                break

        # Convert tokens to words
        caption = [idx2word[idx] for idx in sampled_ids
                  if idx not in [word2idx['<start>'], word2idx['<end>'], word2idx['<pad>']]]

    return ' '.join(caption)
