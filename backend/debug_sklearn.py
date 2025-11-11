import joblib
import pickle
import os

BASE_DIR = os.getcwd()

sklearn_files = {
    'sklearn_logreg': 'models/log_reg/logreg_sklearn.pkl',
    'sklearn_nb': 'models/naive_bayes/naive_bayes_sklearn.pkl',
    'sklearn_DT': 'models/decision_tree/sklearn_DT.pkl',
    'sklearn_RF': 'models/random_forest/random_forest_sklearn.pkl',
    'sklearn_GB': 'models/random_forest/gbdt_sklearn.pkl',
    'sklearn_KNN': 'models/KNN/knn_sklearn.pkl',
}

print("Checking sklearn pickle files...")
print("="*60)

for name, rel_path in sklearn_files.items():
    path = os.path.join(BASE_DIR, rel_path)
    print(f"\n{name}:")
    print(f"  Path: {path}")
    
    # Check if file exists
    if not os.path.exists(path):
        print(f"  ✗ File does not exist")
        continue
    
    # Check file size
    size = os.path.getsize(path)
    print(f"  ✓ File exists, size: {size} bytes")
    
    if size == 0:
        print(f"  ✗ File is empty!")
        continue
    
    # Try to load with joblib
    try:
        model = joblib.load(path)
        print(f"  ✓ Loaded with joblib successfully")
        print(f"     Type: {type(model)}")
    except EOFError as e:
        print(f"  ✗ EOFError with joblib: {e}")
        
        # Try with regular pickle
        try:
            with open(path, 'rb') as f:
                model = pickle.load(f)
            print(f"  ✓ Loaded with pickle successfully")
            print(f"     Type: {type(model)}")
        except Exception as e2:
            print(f"  ✗ Failed with pickle too: {e2}")
    except Exception as e:
        print(f"  ✗ Error: {type(e).__name__}: {e}")

print("\n" + "="*60)