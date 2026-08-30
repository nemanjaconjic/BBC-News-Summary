from pathlib import Path
import random, time, mlflow, numpy as np, pandas as pd, torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from embedding_model import EmbeddingClassifier

SEED, BATCH, EPOCHS, LR, FOLDS = 42, 128, 10, 1e-3, 3
HIDDEN, DROPOUT = 512, 0.5

def seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def evaluate(model, loader, loss_fn):
    model.eval(); loss=0; ys=[]; ps=[]; probs=[]
    with torch.no_grad():
        for x,y in loader:
            z=model(x); loss+=loss_fn(z,y).item(); p=torch.softmax(z,1)[:,1]
            ys.extend(y.numpy()); ps.extend(z.argmax(1).numpy()); probs.extend(p.numpy())
    return loss/len(loader), accuracy_score(ys,ps), precision_score(ys,ps,zero_division=0), recall_score(ys,ps,zero_division=0), f1_score(ys,ps,zero_division=0), ys, ps, probs

def main():
    seed(SEED)
    root=Path(__file__).resolve().parent.parent
    data=torch.load(root/"data/processed/distilbert_embeddings.pt",weights_only=False)
    X=data["embeddings"].float(); y=data["labels"].long()
    groups=np.array([f"{c}_{f}" for c,f in zip(data["category"],data["file_name"])])
    mlflow.set_tracking_uri(f"sqlite:///{(root/'mlflow.db').resolve().as_posix()}")
    mlflow.set_experiment("BBC-News-Summary-Transformer")
    out=root/"results/metrics"; models=root/"results/models"
    out.mkdir(parents=True,exist_ok=True); models.mkdir(parents=True,exist_ok=True)
    results=[]; history=[]; oof=[]

    for fold,(tr,va) in enumerate(GroupKFold(FOLDS).split(X,y,groups),1):
        seed(SEED+fold)
        train=DataLoader(TensorDataset(X[tr],y[tr]),batch_size=BATCH,shuffle=True)
        val=DataLoader(TensorDataset(X[va],y[va]),batch_size=BATCH)
        model=EmbeddingClassifier(768,HIDDEN,DROPOUT)
        params=sum(p.numel() for p in model.parameters() if p.requires_grad)
        opt=torch.optim.AdamW(model.parameters(),lr=LR); loss_fn=nn.CrossEntropyLoss()
        path=models/f"transformer3_fold{fold}.pt"; best=None; best_epoch=0; start=time.perf_counter()

        with mlflow.start_run(run_name=f"transformer3_fold_{fold}"):
            mlflow.log_params({"model":"DistilBERT embeddings + MLP","fold":fold,"batch_size":BATCH,"epochs":EPOCHS,"learning_rate":LR,"hidden":HIDDEN,"dropout":DROPOUT,"folds":FOLDS,"seed":SEED+fold,"trainable_parameters":params})

            for epoch in range(1,EPOCHS+1):
                model.train(); total=0; t=time.perf_counter()
                for xb,yb in train:
                    opt.zero_grad(); z=model(xb); loss=loss_fn(z,yb); loss.backward(); opt.step(); total+=loss.item()

                train_loss=total/len(train)
                val_loss,acc,pre,rec,f1,_,_,_=evaluate(model,val,loss_fn)
                row={"fold":fold,"epoch":epoch,"train_loss":train_loss,"val_loss":val_loss,"accuracy":acc,"precision":pre,"recall":rec,"f1":f1,"epoch_time":time.perf_counter()-t}
                history.append(row)
                for k,v in row.items():
                    if k not in ("fold","epoch"): mlflow.log_metric(k,v,step=epoch)
                print(f"Fold {fold} Epoch {epoch}/{EPOCHS} | Acc {acc:.4f} | F1 {f1:.4f} | Loss {val_loss:.4f}")

                if best is None or f1>best[4]:
                    best=(val_loss,acc,pre,rec,f1); best_epoch=epoch; torch.save(model.state_dict(),path)

            train_time=time.perf_counter()-start
            model.load_state_dict(torch.load(path,weights_only=True))
            t=time.perf_counter(); val_loss,acc,pre,rec,f1,ys,ps,probs=evaluate(model,val,loss_fn); inf=time.perf_counter()-t
            size=path.stat().st_size/1024**2
            results.append({"fold":fold,"best_epoch":best_epoch,"accuracy":acc,"precision":pre,"recall":rec,"f1":f1,"val_loss":val_loss,"training_time":train_time,"inference_time":inf,"model_size_mb":size,"trainable_parameters":params})
            oof.extend({"fold":fold,"label":a,"prediction":b,"probability":c} for a,b,c in zip(ys,ps,probs))
            mlflow.log_metrics({"best_accuracy":acc,"best_precision":pre,"best_recall":rec,"best_f1":f1,"training_time":train_time,"inference_time":inf,"model_size_mb":size})

    r=pd.DataFrame(results); h=pd.DataFrame(history); p=pd.DataFrame(oof)
    r.to_csv(out/"transformer3_cv_results.csv",index=False)
    h.to_csv(out/"transformer3_history.csv",index=False)
    p.to_csv(out/"transformer3_oof_predictions.csv",index=False)

    print("\nTRANSFORMER 3")
    print(r[["fold","accuracy","precision","recall","f1","trainable_parameters"]].to_string(index=False))
    for m in ["accuracy","precision","recall","f1"]: print(f"{m}: {r[m].mean():.4f} ± {r[m].std():.4f}")
    print(f"Average training time: {r.training_time.mean():.2f}s")
    print(f"Trainable parameters: {int(r.trainable_parameters.iloc[0]):,}")

if __name__=="__main__":
    main()