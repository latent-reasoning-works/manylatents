#!/bin/bash
# Setup script for manylatents extensions
# This script helps you install optional extensions for domain-specific functionality

set -e

EXTENSIONS_DIR="extensions"

echo "=========================================================================="
echo "📦 manylatents Extensions Setup"
echo "=========================================================================="
echo ""
echo "This script will help you install optional extensions for manylatents."
echo ""

# Function to install extension
install_extension() {
    local name=$1
    local repo=$2
    local description=$3
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📦 $name"
    echo "   $description"
    echo "   Repository: $repo"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    read -p "Install $name? [y/N] " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Installing $name..."
        
        # Try to install from git directly
        if pip install "git+$repo"; then
            echo "✅ $name installed successfully"
        else
            echo "⚠️  Direct installation failed. Trying local clone..."
            
            # Create extensions directory if needed
            mkdir -p "$EXTENSIONS_DIR"
            
            # Clone to extensions directory
            local dir_name=$(basename "$repo" .git)
            if [ ! -d "$EXTENSIONS_DIR/$dir_name" ]; then
                git clone "$repo" "$EXTENSIONS_DIR/$dir_name"
            else
                echo "Already cloned. Pulling latest changes..."
                (cd "$EXTENSIONS_DIR/$dir_name" && git pull)
            fi
            
            # Install in editable mode
            pip install -e "$EXTENSIONS_DIR/$dir_name"
            echo "✅ $name installed successfully from local clone"
        fi
    else
        echo "⏭️  Skipping $name"
    fi
    echo ""
}

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "❌ Error: pip not found. Please install pip first."
    exit 1
fi

echo "Available extensions:"
echo ""

# manylatents-omics
install_extension \
    "manylatents-omics" \
    "https://github.com/latent-reasoning-works/manylatents-omics.git" \
    "🧬 Genetics and population genetics support (PLINK datasets, geographic metrics, etc.)"

# Add more extensions here as they become available
# install_extension \
#     "manylatents-imaging" \
#     "https://github.com/latent-reasoning-works/manylatents-imaging.git" \
#     "🖼️  Medical imaging support"

echo "=========================================================================="
echo "✅ Setup complete!"
echo "=========================================================================="
echo ""
echo "To verify your extensions are working, run:"
echo ""
echo "  python -c 'from manylatents.omics.data import PlinkDataset; print(\"✅ omics works!\")'"
echo ""
echo "See EXTENSIONS.md for usage documentation."
