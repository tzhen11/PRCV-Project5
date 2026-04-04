
"""
Aafi Mansuri, Terry Zhen
Apr 2026
CS 5330 - Project 5: Recognition using Deep Networks

Vision Transformer for MNIST digit recognition
Based on template by Bruce A. Maxwell and Andy Zhao
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from mnist_network import load_data


class NetConfig:

    def __init__(self,
                 name='vit_base',
                 dataset='mnist',
                 patch_size=4,
                 stride=2,
                 embed_dim=48,
                 depth=4,
                 num_heads=8,
                 mlp_dim=128,
                 dropout=0.1,
                 use_cls_token=False,
                 epochs=15,
                 batch_size=64,
                 lr=1e-3,
                 weight_decay=1e-4,
                 seed=0,
                 optimizer='adamw',
                 device='cpu',
                 ):

        # Dataset fixed attributes
        self.image_size = 28
        self.in_channels = 1
        self.num_classes = 10

        # Variable parameters
        self.name = name
        self.dataset = dataset
        self.patch_size = patch_size
        self.stride = stride
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.use_cls_token = use_cls_token
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.optimizer = optimizer
        self.device = device

        s = "Name,Dataset,PatchSize,Stride,Dim,Depth,Heads,MLPDim,Dropout,CLS,Epochs,Batch,LR,Decay,Seed,Optimizer,TestAcc,BestEpoch\n"
        s += "%s,%s,%d,%d,%d,%d,%d,%d,%0.2f,%s,%d,%d,%f,%f,%d,%s," % (
            self.name, self.dataset, self.patch_size, self.stride,
            self.embed_dim, self.depth, self.num_heads, self.mlp_dim,
            self.dropout, self.use_cls_token, self.epochs, self.batch_size,
            self.lr, self.weight_decay, self.seed, self.optimizer
        )
        self.config_string = s


# Patch Embedding: converts image into sequence of patch tokens
class PatchEmbedding(nn.Module):
    """
    Converts an image into a sequence of patch embeddings.
    Input: (B, C, H, W) -> Output: (B, N, D)
    """

    def __init__(self, image_size, patch_size, stride, in_channels, embed_dim):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.stride = stride
        self.embed_dim = embed_dim

        # Unfold extracts sliding patches from the image
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=stride)

        # Each patch flattened is in_channels * patch_size * patch_size
        self.patch_dim = in_channels * patch_size * patch_size

        # Project flattened patch into embedding space
        self.proj = nn.Linear(self.patch_dim, self.embed_dim)

        self.num_patches = self._compute_num_patches()

    # Compute number of patches along each dimension
    def _compute_num_patches(self):
        positions_per_dim = ((self.image_size - self.patch_size) // self.stride) + 1
        return positions_per_dim * positions_per_dim

    # Extract patches, transpose, and project to embeddings
    def forward(self, x):
        x = self.unfold(x)           # (B, patch_dim, N)
        x = x.transpose(1, 2)        # (B, N, patch_dim)
        x = self.proj(x)             # (B, N, embed_dim)
        return x    

# Vision Transformer network for MNIST classification
class NetTransformer(nn.Module):

    def __init__(self, config):
        super(NetTransformer, self).__init__()

        # Patch embedding layer
        self.patch_embed = PatchEmbedding(
            image_size=config.image_size,
            patch_size=config.patch_size,
            stride=config.stride,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )

        num_tokens = self.patch_embed.num_patches
        print(f"Number of tokens: {num_tokens}")

        # Optional CLS token for classification
        self.use_cls_token = config.use_cls_token
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
            total_tokens = num_tokens + 1
        else:
            self.cls_token = None
            total_tokens = num_tokens

        # Learnable positional embedding for each token
        self.pos_embed = nn.Parameter(torch.zeros(1, total_tokens, config.embed_dim))
        self.pos_dropout = nn.Dropout(config.dropout)

        # Transformer encoder: stack of self-attention + feedforward layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.mlp_dim,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.depth,
        )

        # Final layer norm before classification
        self.norm = nn.LayerNorm(config.embed_dim)

        # Classification head: linear -> GELU -> linear
        self.classifier = nn.Sequential(
            nn.Linear(config.embed_dim, config.mlp_dim),
            nn.GELU(),
            nn.Linear(config.mlp_dim, config.num_classes)
        )

    # Initialize positional embeddings and CLS token
    def _init_parameters(self):
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    # Forward pass: patches -> tokens -> transformer -> classify
    def forward(self, x):
        # Convert image to patch embeddings: (B, 1, 28, 28) -> (B, N, D)
        x = self.patch_embed(x)

        batch_size = x.size(0)

        # Prepend CLS token if used
        if self.use_cls_token:
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_token, x], dim=1)

        # Add positional embedding so tokens know their spatial location
        x = x + self.pos_embed

        # Dropout after embedding
        x = self.pos_dropout(x)

        # Run through transformer encoder layers
        x = self.encoder(x)

        # Pool tokens into single representation
        if self.use_cls_token:
            x = x[:, 0]        # Use CLS token
        else:
            x = x.mean(dim=1)  # Average all tokens

        # Final normalization
        x = self.norm(x)

        # Classification
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)


# Train transformer for one epoch
def train_epoch(model, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to('cuda'), target.to('cuda')
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += len(data)

        if batch_idx % 200 == 0:
            print(f"  Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] "
                  f"Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


# Evaluate transformer on a data loader
def evaluate(model, loader):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to('cuda'), target.to('cuda')
            output = model(data)
            total_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = 100.0 * correct / len(loader.dataset)

    return avg_loss, accuracy


# Plot training and test curves
def plot_curves(train_losses, test_losses, train_accs, test_accs):
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, 'b-o', label='Train')
    ax1.plot(epochs, test_losses, 'r-o', label='Test')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, train_accs, 'b-o', label='Train')
    ax2.plot(epochs, test_accs, 'r-o', label='Test')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.suptitle('Transformer MNIST Training')
    plt.tight_layout()
    plt.savefig('transformer_training_curves.png', dpi=150)
    plt.show()
    print("Saved transformer_training_curves.png")


"""
Main function:
    Builds, trains, and evaluates Vision Transformer on MNIST with default config settings
"""
def main(argv):
    print(torch.cuda.is_available())

    # Use default config
    config = NetConfig()
    print(f"Config: patch={config.patch_size}, stride={config.stride}, "
          f"dim={config.embed_dim}, depth={config.depth}, heads={config.num_heads}")

    # Load MNIST data
    train_loader, test_loader = load_data(config.batch_size)

    # Build transformer model
    model = NetTransformer(config)
    model.to('cuda')
    model._init_parameters()
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # AdamW optimizer as specified in config
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, epoch)
        test_loss, test_acc = evaluate(model, test_loader)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f} Acc={train_acc:.1f}% | "
              f"Test Loss={test_loss:.4f} Acc={test_acc:.1f}%")

    plot_curves(train_losses, test_losses, train_accs, test_accs)

    torch.save(model.state_dict(), 'transformer_model.pth')
    print("Model saved to 'transformer_model.pth'")


if __name__ == "__main__":
    main(sys.argv)