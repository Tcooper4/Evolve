import inspect
import sys

sys.path.insert(0, ".")
try:
    from trading.analysis import ml_score_trainer
    print("Import OK")
    # Check train() signature
    sig = inspect.signature(
        ml_score_trainer.train
        if hasattr(ml_score_trainer, "train")
        else ml_score_trainer.MLScoreTrainer().train
    )
    print(f"train() sig: {sig}")
    print("Trainer importable: OK")
except Exception as e:
    print(f"Import FAILED: {e}")
    import traceback

    traceback.print_exc()
