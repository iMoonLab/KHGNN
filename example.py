#!/usr/bin/env python3
"""
Example usage of KHGNN model
Run this script to see a basic example of how to use the KHGNN model.
"""

import torch
import torch.nn.functional as F
from dhg import Hypergraph

from khgnn_model import KerHGNN
from utils import load_data


def example_usage():
    """Demonstrate basic KHGNN usage"""
    print("🚀 KHGNN Example Usage")
    print("=" * 50)

    # Load a small dataset for demonstration
    print("📊 Loading dataset...")
    try:
        data_name = "cora"  # You can change this to other datasets
        data, edge_list = load_data(data_name)

        print(f"✅ Loaded {data_name} dataset")
        print(f"   - Nodes: {data['features'].shape[0]}")
        print(f"   - Edges: {len(edge_list)}")
        print(f"   - Features: {data['features'].shape[1]}")
        print(f"   - Classes: {data['labels'].max().item() + 1}")

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("💡 Make sure you have the required datasets available")
        return

    # Initialize the model
    print("\n🔧 Initializing KHGNN model...")
    model = KerHGNN(
        in_channels=data["features"].shape[1],
        hid_channels=64,
        num_classes=data["labels"].max().item() + 1,
        num_layer=2,
        kernel_type="poly",  # Try 'poly', 'apoly', or 'mean'
        p_min=-0.5,
        p_max=2.0,
        drop_rate=0.5,
    )

    print(
        f"✅ Model initialized with {sum(p.numel() for p in model.parameters())} parameters"
    )

    # Prepare data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    model = model.to(device)
    features = data["features"].to(device)
    labels = data["labels"].to(device)
    hg = Hypergraph(data["num_vertices"], edge_list)

    # Forward pass
    print("\n⚡ Running forward pass...")
    model.eval()
    with torch.no_grad():
        output = model(features, hg)
        probs = F.softmax(output, dim=1)

    print(f"✅ Forward pass completed")
    print(f"   - Output shape: {output.shape}")
    print(f"   - Prediction probabilities shape: {probs.shape}")

    # Show predictions for first few nodes
    print("\n📈 Sample predictions:")
    for i in range(min(5, len(labels))):
        pred_class = probs[i].argmax().item()
        true_class = labels[i].item()
        confidence = probs[i].max().item()

        status = "✅" if pred_class == true_class else "❌"
        print(
            f"   Node {i}: Pred={pred_class}, True={true_class}, Conf={confidence:.3f} {status}"
        )

    print("\n🎉 Example completed successfully!")
    print("\n💡 Next steps:")
    print("   - Run 'python trans_train.py' for full training")
    print("   - Modify config files to experiment with different settings")
    print("   - Try different kernel types: 'poly', 'apoly', 'mean'")


if __name__ == "__main__":
    example_usage()
