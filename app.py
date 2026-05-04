# =============================================================
#  app.py – Application Flask – TP Deep Learning ENSY
# =============================================================
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os
import base64

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = Flask(__name__)
MODELS = {}

NUM_COLS = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate',
            'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

CAT_COLS = ["job","marital","education","default","housing","loan",
            "contact","month","day_of_week","poutcome"]

def load_models():
    try:
        for name, path in [("cart","models/best_cart.pkl"),
                           ("rf",  "models/best_rf.pkl"),
                           ("gb",  "models/best_gb.pkl")]:
            if os.path.exists(path):
                MODELS[name] = joblib.load(path)
                print(f"  ✔ {name}")

        for key, path in [("feature_names",  "models/feature_names.pkl"),
                          ("top10_features", "models/top10_features.pkl"),
                          ("scaler",         "models/scaler.pkl")]:
            if os.path.exists(path):
                MODELS[key] = joblib.load(path)

        if os.path.exists("models/telemarketing.pkl"):
            MODELS["ann_meta"] = joblib.load("models/telemarketing.pkl")

        if os.path.exists("models/telemarketing.keras"):
            try:
                import tensorflow as tf
                MODELS["ann"] = tf.keras.models.load_model("models/telemarketing.keras")
                print("  ✔ ANN chargé")
            except Exception as e:
                print(f"  ⚠ ANN: {e}")

        if os.path.exists("models/best_fashion.keras"):
            try:
                import tensorflow as tf
                MODELS["fashion"] = tf.keras.models.load_model("models/best_fashion.keras")
                print("  ✔ Fashion chargé")
            except Exception as e:
                print(f"  ⚠ Fashion: {e}")

        print("✅ Modèles chargés.")
    except Exception as e:
        print(f"⚠ Erreur : {e}")

load_models()

FASHION_CLASSES = ["T-shirt/top","Pantalon","Pull","Robe","Manteau",
                   "Sandale","Chemise","Sneaker","Sac","Bottine"]


def build_full_vector(features_dict, feature_names, scaler):
    """Construit le vecteur complet de 46 features scalé → shape (1, 46)."""
    num_vals = {}
    for col in NUM_COLS:
        try:
            num_vals[col] = float(features_dict.get(col, 0))
        except:
            num_vals[col] = 0.0

    X_num = pd.DataFrame([num_vals])[NUM_COLS].astype(float)
    X_num_scaled = scaler.transform(X_num)[0]

    X_full = np.zeros(len(feature_names), dtype=float)

    for i, col in enumerate(NUM_COLS):
        if col in feature_names:
            X_full[feature_names.index(col)] = X_num_scaled[i]

    for col, val in features_dict.items():
        if col in CAT_COLS:
            dummy_col = f"{col}_{val}"
            if dummy_col in feature_names:
                X_full[feature_names.index(dummy_col)] = 1.0

    return X_full.reshape(1, -1).astype(np.float32)


def build_input_dataframe(features_dict, feature_names):
    """Pour Partie 1 (sklearn) — sans scaler."""
    cat_present = [c for c in CAT_COLS if c in features_dict]
    num_data = {}
    cat_data = {}
    for k, v in features_dict.items():
        if k in CAT_COLS:
            cat_data[k] = v
        else:
            try:
                num_data[k] = float(v)
            except:
                num_data[k] = 0.0

    df = pd.DataFrame([{**num_data, **cat_data}])
    if cat_present:
        df = pd.get_dummies(df, columns=cat_present, drop_first=True)
    for col in df.select_dtypes(include='bool').columns:
        df[col] = df[col].astype(int)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    return df[feature_names].astype(float)


def get_images(part_num):
    """Charge les images depuis figures/partieN/ dans le repo."""
    images = []
    fig_path = f"figures/partie{part_num}"
    if not os.path.exists(fig_path):
        return images
    for fname in sorted(os.listdir(fig_path)):
        if fname.lower().endswith((".png", ".jpg")):
            try:
                with open(os.path.join(fig_path, fname), "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                label = fname.rsplit(".", 1)[0]
                label = "_".join(label.split("_")[1:]).replace("_", " ").title()
                images.append({"name": label,
                                "data": f"data:image/png;base64,{b64}"})
            except:
                pass
    return images


# =============================================================
#  ROUTES
# =============================================================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/partie1")
def partie1():
    return render_template("partie1.html", images=get_images(1))

@app.route("/partie2")
def partie2():
    return render_template("partie2.html", images=get_images(2))

@app.route("/partie3")
def partie3():
    return render_template("partie3.html", classes=FASHION_CLASSES,
                           images=get_images(3))


# =============================================================
#  APIs
# =============================================================

@app.route("/api/predict_p1", methods=["POST"])
def predict_p1():
    try:
        data          = request.get_json()
        model         = MODELS.get(data.get("model", "rf"))
        if not model:
            return jsonify({"error": "Modèle non disponible"}), 400
        feature_names = MODELS.get("feature_names", [])
        X_df = build_input_dataframe(data.get("features", {}), feature_names)
        prob = float(model.predict_proba(X_df)[0][1])
        pred = int(prob >= 0.5)
        return jsonify({
            "prediction":  pred,
            "label":       "Souscription" if pred else "Pas de souscription",
            "probability": round(prob * 100, 2),
            "model":       data.get("model", "rf").upper()
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict_p2", methods=["POST"])
def predict_p2():
    try:
        data          = request.get_json()
        ann           = MODELS.get("ann")
        if not ann:
            return jsonify({"error": "Modèle ANN non disponible"}), 400
        scaler        = MODELS.get("scaler")
        feature_names = MODELS.get("feature_names", [])
        features      = data.get("features", {})

        X_full = build_full_vector(features, feature_names, scaler)

        prob = float(ann.predict(X_full, verbose=0)[0][0])
        pred = int(prob >= 0.5)
        return jsonify({
            "prediction":  pred,
            "label":       "Souscription" if pred else "Pas de souscription",
            "probability": round(prob * 100, 2),
            "model":       MODELS.get("ann_meta", {}).get("model_name", "ANN")
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict_p3", methods=["POST"])
def predict_p3():
    try:
        data    = request.get_json()
        pixels  = data.get("pixels", [])
        if len(pixels) != 784:
            return jsonify({"error": "784 pixels attendus"}), 400
        fashion = MODELS.get("fashion")
        if not fashion:
            return jsonify({"error": "Modèle Fashion non disponible"}), 400
        X     = np.array(pixels, dtype=np.float32).reshape(1, 28, 28, 1) / 255.0
        probs = fashion.predict(X, verbose=0)[0]
        pred  = int(np.argmax(probs))
        top3  = sorted(enumerate(probs), key=lambda x: -x[1])[:3]
        return jsonify({
            "prediction": pred,
            "label":      FASHION_CLASSES[pred],
            "confidence": round(float(probs[pred]) * 100, 2),
            "top3": [{"class": FASHION_CLASSES[i], "prob": round(float(p)*100, 2)}
                     for i, p in top3]
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)