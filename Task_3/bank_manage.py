# setup_and_run.py
# Setup script for Bank Marketing Decision Tree Classifier

import subprocess
import sys
import importlib

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def check_and_install_dependencies():
    """Check and install required packages"""
    required_packages = [
        'ucimlrepo',
        'pandas',
        'numpy', 
        'scikit-learn',
        'matplotlib',
        'seaborn'
    ]
    
    print("🔍 Checking dependencies...")
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'scikit-learn':
                importlib.import_module('sklearn')
            else:
                importlib.import_module(package)
            print(f"✅ {package} is already installed")
        except ImportError:
            print(f"❌ {package} is missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        for package in missing_packages:
            if not install_package(package):
                print(f"❌ Failed to install {package}. Please install manually:")
                print(f"   pip install {package}")
                return False
        print("✅ All dependencies installed successfully!")
    else:
        print("✅ All dependencies are already installed!")
    
    return True

def run_bank_marketing_classifier():
    """Run the main bank marketing classifier"""
    print("\n" + "="*60)
    print("🚀 RUNNING BANK MARKETING CLASSIFIER")
    print("="*60)
    
    try:
        # Import and run the main classifier
        exec(open('bank_marketing_classifier.py').read())
    except FileNotFoundError:
        print("❌ bank_marketing_classifier.py not found!")
        print("Please ensure the main classifier file is in the same directory.")
    except Exception as e:
        print(f"❌ Error running classifier: {e}")

def main():
    """Main setup and run function"""
    print("🏦 Bank Marketing Decision Tree Classifier Setup")
    print("="*60)
    
    # Step 1: Check and install dependencies
    if not check_and_install_dependencies():
        print("❌ Setup failed. Please install missing dependencies manually.")
        return
    
    # Step 2: Run the classifier
    print("\n" + "="*60)
    print("✅ Setup complete! Starting classifier...")
    print("="*60)
    
    # Import all necessary modules for the classifier
    try:
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
        from sklearn.preprocessing import OneHotEncoder, LabelEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.tree import DecisionTreeClassifier, plot_tree
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
        import matplotlib.pyplot as plt
        import seaborn as sns
        from ucimlrepo import fetch_ucirepo
        import warnings
        warnings.filterwarnings('ignore')
        
        # Now run the complete classifier pipeline
        print("📥 Fetching Bank Marketing dataset from UCI repository...")
        
        # Fetch dataset (ID 222 is the Bank Marketing dataset)
        bank_marketing = fetch_ucirepo(id=222)
        
        # Get data as pandas dataframes
        X = bank_marketing.data.features
        y = bank_marketing.data.targets
        
        # Combine features and target into one DataFrame
        df = pd.concat([X, y], axis=1)
        
        print("✅ Successfully loaded dataset from UCI repository")
        print(f"✅ Dataset shape: {df.shape}")
        
        # Display metadata
        print("\n📋 Dataset Metadata:")
        print(f"Name: {bank_marketing.metadata.get('name', 'N/A')}")
        print(f"Description: {bank_marketing.metadata.get('abstract', 'N/A')[:200]}...")
        
        # Quick dataset exploration
        print("\n📊 Dataset Overview:")
        print(f"Features: {X.shape[1]}")
        print(f"Samples: {X.shape[0]}")
        print(f"Target variable: {y.columns[0]}")
        print(f"Target distribution:\n{y.iloc[:,0].value_counts()}")
        
        # Prepare data for modeling
        print("\n🔧 Preprocessing data...")
        
        # Encode target variable
        le = LabelEncoder()
        y_encoded = le.fit_transform(y.iloc[:, 0])
        
        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"Categorical features: {len(categorical_cols)}")
        print(f"Numerical features: {len(numerical_cols)}")
        
        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols)
            ]
        )
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Create and train model
        print("\n🤖 Training Decision Tree Classifier...")
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier(
                max_depth=10, 
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ))
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        y_test_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        roc_auc = roc_auc_score(y_test, y_test_proba)
        
        print("\n📈 Model Performance:")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"ROC AUC Score: {roc_auc:.4f}")
        
        print("\n📊 Detailed Classification Report:")
        print(classification_report(y_test, y_test_pred, target_names=le.classes_))
        
        # Feature importance
        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
        feature_importance = pipeline.named_steps['classifier'].feature_importances_
        
        # Top 10 most important features
        top_indices = np.argsort(feature_importance)[::-1][:10]
        print("\n🎯 Top 10 Most Important Features:")
        for i, idx in enumerate(top_indices, 1):
            print(f"{i:2d}. {feature_names[idx]}: {feature_importance[idx]:.4f}")
        
        # Create simple visualization
        print("\n📊 Creating visualization...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        axes[1].plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend(loc="lower right")
        
        plt.tight_layout()
        plt.savefig('quick_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save predictions
        predictions_df = X_test.copy().reset_index(drop=True)
        predictions_df['actual'] = y_test
        predictions_df['predicted'] = y_test_pred
        predictions_df['probability'] = y_test_proba
        predictions_df.to_csv('predictions.csv', index=False)
        
        print("\n✅ Analysis complete!")
        print("📁 Files generated:")
        print("   - quick_results.png (confusion matrix & ROC curve)")
        print("   - predictions.csv (test set predictions)")
        
        print(f"\n🎯 Summary:")
        print(f"   - Accuracy: {test_accuracy:.1%}")
        print(f"   - ROC AUC: {roc_auc:.3f}")
        print(f"   - Model successfully predicts bank marketing outcomes!")
        
    except Exception as e:
        print(f"❌ Error running classifier: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()