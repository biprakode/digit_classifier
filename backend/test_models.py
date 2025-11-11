try:
    from models import models as model_classes
    
    print(f"\n✓ Successfully imported models")
    print(f"  Found {len(model_classes)} model classes\n")
    
    print("Model classes found:")
    print("="*60)
    for name in model_classes.keys():
        print(f"  • {name}")
    print("="*60)
    
    # Try instantiating one model
    print("\nTrying to instantiate sklearn_logreg...")
    try:
        model = model_classes['sklearn_logreg']()
        print("✓ Successfully instantiated sklearn_logreg")
    except Exception as e:
        print(f"✗ Failed to instantiate: {e}")
        
except ImportError as e:
    print(f"✗ Import error: {e}")
except Exception as e:
    print(f"✗ Error: {e}")