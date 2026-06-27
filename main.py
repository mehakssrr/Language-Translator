pip install torch torchvision
!python main.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Define the Encoder model
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input_seq, hidden):
        # input_seq is (seq_len, batch_size) coming from Seq2Seq.forward
        embedded = self.embedding(input_seq) # results in (seq_len, batch_size, hidden_size)
        # GRU expects input of shape (seq_len, batch, features) by default (batch_first=False)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

# Define the Decoder model
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_seq, hidden):
        # input_seq is (1, batch_size) coming from Seq2Seq.forward
        embedded = self.embedding(input_seq) # results in (1, batch_size, hidden_size)
        # GRU expects input of shape (seq_len, batch, features) by default (batch_first=False)
        output, hidden = self.gru(embedded, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

# Define the Seq2Seq model that combines the Encoder and Decoder
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, input_seq, target_seq, teacher_forcing_ratio=0.5):
        # input_seq: (batch_size, seq_len), target_seq: (batch_size, seq_len) after collate_fn

        batch_size = input_seq.size(0)
        target_len = target_seq.size(1) # Max target sequence length in batch
        target_vocab_size = self.decoder.out.out_features

        # Prepare outputs tensor. It should be (target_len, batch_size, target_vocab_size)
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)

        # Initial hidden state for encoder
        # Encoder expects (seq_len, batch_size) for input, hidden (1, batch_size, hidden_size)
        # Permute input_seq to (seq_len, batch_size) for encoder
        initial_encoder_hidden = torch.zeros(1, batch_size, self.encoder.hidden_size).to(self.device)
        encoder_output, hidden = self.encoder(input_seq.permute(1, 0), initial_encoder_hidden)

        # First input to the decoder is the <SOS> token for each sequence in the batch
        decoder_input = torch.tensor([SOS_token] * batch_size).unsqueeze(0).to(self.device) # (1, batch_size)

        for t in range(1, target_len):
            # Decoder input should be (1, batch_size) for a single time step
            output, hidden = self.decoder(decoder_input, hidden)
            outputs[t] = output # outputs[t] is (batch_size, target_vocab_size)

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio # Use .item() for scalar comparison
            top1 = output.max(1)[1] # Get the index of the best word (batch_size)

            # If teacher forcing, use actual target word; else, use decoder's prediction
            decoder_input = target_seq[:, t].unsqueeze(0) if teacher_force else top1.unsqueeze(0) # (1, batch_size)

        return outputs

# Define the dataset class
class TranslationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        return self.data[index]

# Custom collate_fn for DataLoader
def collate_fn_pad_sequences(batch):
    # batch is a list of tuples: [(input_seq_tensor, target_seq_tensor), ...]
    input_sequences = [item[0] for item in batch]
    target_sequences = [item[1] for item in batch]

    # Pad input sequences
    # pad_sequence expects a list of tensors, where each tensor is 1D (seq_len)
    # It returns a tensor of shape (max_seq_len, batch_size) by default, or (batch_size, max_seq_len) if batch_first=True
    padded_input_sequences = pad_sequence(input_sequences, batch_first=True, padding_value=input_vocab['<pad>'])

    # Pad target sequences
    padded_target_sequences = pad_sequence(target_sequences, batch_first=True, padding_value=output_vocab['<pad>'])

    return padded_input_sequences, padded_target_sequences

# Define the training function
def train(model, optimizer, criterion, input_seq, target_seq):
    optimizer.zero_grad()

    input_seq = input_seq.to(device)
    target_seq = target_seq.to(device)

    # model output will be (target_len, batch_size, vocab_size)
    output = model(input_seq, target_seq)

    # The target_seq given to the model starts with <sos>, so the first output is for the first actual target word.
    # We need to reshape output and target_seq for NLLLoss.
    # output = output[1:].view(-1, output.shape[-1]) means skip <sos> output
    # target_seq = target_seq[:, 1:].reshape(-1) means skip <sos> target
    # current model output is (target_len, batch_size, vocab_size), need to reshape to (batch_size * (target_len-1), vocab_size)
    # target_seq is (batch_size, target_len), need to reshape to (batch_size * (target_len-1))

    # output has shape (target_len, batch_size, vocab_size)
    # criterion expects input (N, C) and target (N)
    # We need to flatten the output and target from index 1 (after SOS token)
    output_dim = output.shape[-1]
    output = output[1:].view(-1, output_dim) # ( (target_len-1) * batch_size, vocab_size )
    target_seq = target_seq[:, 1:].reshape(-1) # ( (target_len-1) * batch_size )

    loss = criterion(output, target_seq)
    loss.backward()

    # Clip gradients to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

    optimizer.step()

    return loss.item()

# --- FIX START ---
# Define placeholder variables for demonstration
# In a real scenario, these would be populated from your dataset and vocabulary building process
input_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, 'hello': 3, 'world': 4, 'how': 5, 'are': 6, 'you': 7}
output_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, 'bonjour': 3, 'monde': 4, 'comment': 5, 'allez': 6, 'vous': 7}

# Example dummy data: list of (input_sequence_tensor, target_sequence_tensor)
# Each sequence is a list of token indices
dummy_input_seq = [input_vocab['<sos>'], input_vocab['hello'], input_vocab['world'], input_vocab['<eos>']]
dummy_target_seq = [output_vocab['<sos>'], output_vocab['bonjour'], output_vocab['monde'], output_vocab['<eos>']]

data = [
    (torch.tensor(dummy_input_seq), torch.tensor(dummy_target_seq)),
    (torch.tensor([input_vocab['<sos>'], input_vocab['how'], input_vocab['are'], input_vocab['you'], input_vocab['<eos>']]),
     torch.tensor([output_vocab['<sos>'], output_vocab['comment'], output_vocab['allez'], output_vocab['vous'], output_vocab['<eos>']]))
]

# Define special tokens
SOS_token = output_vocab['<sos>']  # Start of Sequence token index
EOS_token = output_vocab['<eos>']  # End of Sequence token index (not directly causing error here, but good practice)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# --- FIX END ---

# Define the main function
def main():
    # Define hyperparameters
    input_size = len(input_vocab)
    output_size = len(output_vocab)
    hidden_size = 256
    learning_rate = 0.01
    batch_size = 32
    num_epochs = 10
    teacher_forcing_ratio = 0.5

    # Prepare the dataset
    dataset = TranslationDataset(data)
    # Increase batch_size if data is too small to avoid issues with DataLoader
    # or handle the case where batch_size > len(dataset)
    current_batch_size = min(batch_size, len(dataset)) if len(dataset) > 0 else 1 # Ensure batch_size is not larger than dataset size
    # Pass the custom collate_fn to DataLoader
    dataloader = DataLoader(dataset, batch_size=current_batch_size, shuffle=True, collate_fn=collate_fn_pad_sequences)

    # Initialize the models
    encoder = Encoder(input_size, hidden_size)
    decoder = Decoder(hidden_size, output_size)
    model = Seq2Seq(encoder, decoder, device).to(device)

    # Define the loss function and optimizer
    criterion = nn.NLLLoss(ignore_index=output_vocab['<pad>']) # Ignore padding tokens in loss calculation
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        # Handle case where dataloader might be empty if dataset is empty
        if len(dataloader) == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Dataloader is empty, skipping training.")
            continue
        for i, (input_seq, target_seq) in enumerate(dataloader):
            loss = train(model, optimizer, criterion, input_seq, target_seq)
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), 'translation_model.pt')

# Run the main function
if __name__ == '__main__':
    main()
    #the program was done to supervise the overview od the conditoons of the fragments which were then converted to the opotsitoon of the machine learning model then next day they weere the ktaken into the consideration then and there weere the option of the translator which involces the monitory decision
