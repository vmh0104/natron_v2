"""
Main Training Pipeline: End-to-End Natron Training
"""
import pandas as pd
import numpy as np
import yaml
import torch
from sklearn.model_selection import train_test_split
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.feature_engine import FeatureEngine
from src.label_generator import LabelGeneratorV2
from src.sequence_creator import SequenceCreator
from src.model import NatronTransformer, MaskedTokenModel
from src.pretrain import pretrain, PretrainDataset
from src.train import supervised_train, SupervisedDataset
from torch.utils.data import DataLoader


def load_config(config_path: str = 'config/config.yaml'):
    """Load configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training pipeline"""
    print("🚀 Natron Transformer V2 - Training Pipeline")
    print("=" * 60)
    
    # Load config
    config = load_config()
    
    # Device
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"📱 Using device: {device}")
    
    # Step 1: Load data
    print("\n📂 Step 1: Loading data...")
    csv_path = config['data']['csv_path']
    if not os.path.exists(csv_path):
        print(f"❌ Error: Data file not found at {csv_path}")
        print("   Please ensure data_export.csv exists in the data/ directory")
        return
    
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} rows from {csv_path}")
    print(f"   Columns: {list(df.columns)}")
    
    # Step 2: Feature extraction
    print("\n🔧 Step 2: Extracting features...")
    feature_engine = FeatureEngine()
    features_df = feature_engine.fit_transform(df)
    print(f"✅ Generated {len(feature_engine.get_feature_names())} features")
    
    # Step 3: Label generation
    print("\n🏷️  Step 3: Generating labels...")
    label_generator = LabelGeneratorV2(
        buy_threshold=config['labeling']['buy_threshold'],
        sell_threshold=config['labeling']['sell_threshold'],
        neutral_buffer=config['labeling']['neutral_buffer'],
        balance_classes=config['labeling']['balance_classes'],
        stochastic_perturbation=config['labeling']['stochastic_perturbation']
    )
    labels_df = label_generator.generate_labels(features_df)
    
    # Step 4: Create sequences
    print("\n📊 Step 4: Creating sequences...")
    sequence_creator = SequenceCreator(sequence_length=config['data']['sequence_length'])
    X, y = sequence_creator.create_sequences(features_df, labels_df)
    
    # Step 5: Train/Val/Test split
    print("\n✂️  Step 5: Splitting data...")
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=1 - config['data']['train_split'],
        random_state=42
    )
    
    val_size = config['data']['val_split'] / (config['data']['val_split'] + config['data']['test_split'])
    val_idx, test_idx = train_test_split(temp_idx, test_size=1 - val_size, random_state=42)
    
    X_train, y_train = X[train_idx], {k: v[train_idx] for k, v in y.items()}
    X_val, y_val = X[val_idx], {k: v[val_idx] for k, v in y.items()}
    X_test, y_test = X[test_idx], {k: v[test_idx] for k, v in y.items()}
    
    print(f"✅ Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Step 6: Phase 1 - Pretraining
    print("\n🎓 Phase 1: Pretraining (Masked Token + Contrastive Learning)...")
    pretrain_dataset_train = PretrainDataset(X_train)
    pretrain_dataset_val = PretrainDataset(X_val)
    
    pretrain_loader_train = DataLoader(
        pretrain_dataset_train,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    pretrain_loader_val = DataLoader(
        pretrain_dataset_val,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    # Create model
    n_features = X.shape[2]
    encoder = NatronTransformer(
        n_features=n_features,
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_layers=config['model']['num_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        dropout=config['model']['dropout'],
        sequence_length=config['data']['sequence_length']
    )
    
    pretrain_model = MaskedTokenModel(
        encoder=encoder,
        mask_prob=config['training']['pretrain']['mask_prob']
    )
    
    pretrain_path = os.path.join(config['paths']['model_dir'], 'natron_pretrain.pt')
    pretrain(
        model=pretrain_model,
        train_loader=pretrain_loader_train,
        val_loader=pretrain_loader_val,
        device=device,
        num_epochs=config['training']['num_epochs_pretrain'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        reconstruction_weight=config['training']['pretrain']['reconstruction_weight'],
        contrastive_weight=config['training']['pretrain']['contrastive_weight'],
        save_path=pretrain_path
    )
    
    # Load pretrained encoder
    checkpoint = torch.load(pretrain_path, map_location=device)
    encoder.load_state_dict(checkpoint['model_state_dict'])
    print("✅ Pretraining complete!")
    
    # Step 7: Phase 2 - Supervised Fine-Tuning
    print("\n🎯 Phase 2: Supervised Fine-Tuning...")
    supervised_dataset_train = SupervisedDataset(X_train, y_train)
    supervised_dataset_val = SupervisedDataset(X_val, y_val)
    
    supervised_loader_train = DataLoader(
        supervised_dataset_train,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    supervised_loader_val = DataLoader(
        supervised_dataset_val,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    model_path = os.path.join(config['paths']['model_dir'], config['paths']['model_name'])
    supervised_train(
        model=encoder,
        train_loader=supervised_loader_train,
        val_loader=supervised_loader_val,
        device=device,
        num_epochs=config['training']['num_epochs_supervised'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        scheduler_patience=config['training']['supervised']['scheduler_patience'],
        scheduler_factor=config['training']['supervised']['scheduler_factor'],
        early_stopping_patience=config['training']['supervised']['early_stopping_patience'],
        save_path=model_path
    )
    
    print(f"\n🎉 Training complete! Model saved to {model_path}")
    print("\n📋 Next steps:")
    print("   1. Test the model: python src/api.py --model_path model/natron_v2.pt")
    print("   2. Start API server: python src/api.py")
    print("   3. Integrate with MQL5 EA via socket server")


if __name__ == '__main__':
    main()
