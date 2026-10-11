import re
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
DATA_FILE = DATA_DIR / "SMSSpamCollection"
DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\b\d+\b", " NUMBER ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def load_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_FILE.exists():
        archive_path = DATA_DIR / "sms_spam_collection.zip"
        try:
            urllib.request.urlretrieve(DATA_URL, archive_path)
            with zipfile.ZipFile(archive_path) as archive:
                member = next(n for n in archive.namelist() if n.endswith("SMSSpamCollection"))
                with archive.open(member) as src, open(DATA_FILE, "wb") as dst:
                    dst.write(src.read())
        except Exception as exc:
            raise RuntimeError("Dataset download failed. Ensure outbound internet access or place the extracted SMSSpamCollection file in backend/data/.") from exc
    df = pd.read_csv(DATA_FILE, sep="\t", header=None, names=["label", "message"], encoding="utf-8")
    df = df.dropna(subset=["label", "message"])
    df["label"] = df["label"].astype(str).str.lower().map({"ham": 0, "spam": 1})
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df["message"] = df["message"].astype(str)
    if len(df) < 100: raise ValueError("Dataset looks too small. Check backend/data/SMSSpamCollection.")
    return df

def load_or_train_model():
    df=load_dataset()
    X_train,X_test,y_train,y_test=train_test_split(df.message,df.label,test_size=.2,random_state=42,stratify=df.label)
    model=Pipeline([
        ("tfidf",TfidfVectorizer(preprocessor=clean_text,ngram_range=(1,2),min_df=2,sublinear_tf=True)),
        ("classifier",LogisticRegression(max_iter=1000,class_weight="balanced",random_state=42)),
    ])
    model.fit(X_train,y_train)
    pred=model.predict(X_test)
    report=classification_report(y_test,pred,labels=[0,1],target_names=["Not Spam","Spam"],output_dict=True,zero_division=0)
    metrics={
        "total_messages":int(len(df)),"train_messages":int(len(X_train)),"test_messages":int(len(X_test)),
        "spam_messages":int((df.label==1).sum()),"not_spam_messages":int((df.label==0).sum()),
        "accuracy":float(accuracy_score(y_test,pred)),"precision_spam":float(precision_score(y_test,pred,zero_division=0)),
        "recall_spam":float(recall_score(y_test,pred,zero_division=0)),"f1_spam":float(f1_score(y_test,pred,zero_division=0)),
        "confusion_matrix":confusion_matrix(y_test,pred,labels=[0,1]).tolist(),"classification_report":report,
        "confusion_matrix_labels_order":["Not Spam","Spam"]
    }
    return model,metrics,"UCI SMS Spam Collection"

def predict_messages(model,messages):
    probabilities=model.predict_proba(messages)
    predictions=model.predict(messages)
    results=[]
    for pred,p in zip(predictions,probabilities):
        pred=int(pred)
        results.append({"label":"Spam" if pred==1 else "Not Spam","confidence":float(p[pred]),
                        "spam_probability":float(p[1]),"not_spam_probability":float(p[0])})
    return results
