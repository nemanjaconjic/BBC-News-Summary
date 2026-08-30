from pathlib import Path
import pandas as pd

def main():
    root=Path(__file__).resolve().parent.parent
    rows=[]
    for i in range(1,6):
        df=pd.read_csv(root/f"results/metrics/transformer{i}_cv_results.csv")
        rows.append({
            "model":f"Transformer {i}",
            "accuracy":df.accuracy.mean(),
            "precision":df.precision.mean(),
            "recall":df.recall.mean(),
            "f1":df.f1.mean(),
            "training_time":df.training_time.mean(),
            "inference_time":df.inference_time.mean(),
            "model_size_mb":df.model_size_mb.mean(),
            "trainable_parameters":df.trainable_parameters.iloc[0] if "trainable_parameters" in df else None
        })

    result=pd.DataFrame(rows).sort_values("f1",ascending=False)
    out=root/"results/metrics/transformer_comparison.csv"
    result.to_csv(out,index=False)
    print(result.to_string(index=False))
    print(f"\nBest model: {result.iloc[0]['model']} | F1: {result.iloc[0]['f1']:.4f}")

if __name__=="__main__":
    main()