from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

def savefig(path):
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()

def main():
    root=Path(__file__).resolve().parent.parent
    metrics=root/"results/metrics"
    plots=root/"results/plots"
    plots.mkdir(parents=True,exist_ok=True)

    comp=pd.read_csv(metrics/"transformer_comparison.csv").sort_values("model")

    plt.figure(figsize=(8,5))
    plt.bar(comp.model,comp.f1)
    plt.ylabel("F1")
    plt.xticks(rotation=20)
    savefig(plots/"transformer_f1_comparison.png")

    m=comp.set_index("model")[["accuracy","precision","recall","f1"]]
    m.plot(kind="bar",figsize=(10,6))
    plt.ylabel("Score")
    plt.xticks(rotation=20)
    savefig(plots/"transformer_metrics_comparison.png")

    plt.figure(figsize=(9,6))
    for i in range(1,6):
        h=pd.read_csv(metrics/f"transformer{i}_history.csv")
        x=h.groupby("epoch").val_loss.mean()
        plt.plot(x.index,x.values,label=f"Transformer {i}")
    plt.xlabel("Epoch"); plt.ylabel("Validation loss"); plt.legend()
    savefig(plots/"transformer_learning_curves.png")

    best=pd.read_csv(metrics/"transformer2_oof_predictions.csv")
    cm=confusion_matrix(best.label,best.prediction)
    plt.figure(figsize=(5,5))
    plt.imshow(cm)
    plt.xticks([0,1],["0","1"]); plt.yticks([0,1],["0","1"])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    for i in range(2):
        for j in range(2): plt.text(j,i,cm[i,j],ha="center",va="center")
    savefig(plots/"transformer2_confusion_matrix.png")

    fpr,tpr,_=roc_curve(best.label,best.probability)
    plt.figure(figsize=(6,5))
    plt.plot(fpr,tpr,label=f"AUC={auc(fpr,tpr):.3f}")
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.legend()
    savefig(plots/"transformer2_roc_curve.png")

    p,r,_=precision_recall_curve(best.label,best.probability)
    plt.figure(figsize=(6,5))
    plt.plot(r,p,label=f"AUC={auc(r,p):.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend()
    savefig(plots/"transformer2_pr_curve.png")

    print(f"Plots saved to: {plots}")

if __name__=="__main__":
    main()