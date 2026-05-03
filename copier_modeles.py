import shutil, os

SOURCES = {
    "partie1/outputs/models/best_cart.pkl":      "deploy/models/best_cart.pkl",
    "partie1/outputs/models/best_rf.pkl":        "deploy/models/best_rf.pkl",
    "partie1/outputs/models/best_gb.pkl":        "deploy/models/best_gb.pkl",
    "partie1/outputs/models/feature_names.pkl":  "deploy/models/feature_names.pkl",
    "partie1/outputs/models/top10_features.pkl": "deploy/models/top10_features.pkl",
    "partie1/outputs/models/scaler.pkl":         "deploy/models/scaler.pkl",
    "partie2/outputs/models/telemarketing.pkl":  "deploy/models/telemarketing.pkl",
    "partie2/outputs/models/telemarketing.keras":"deploy/models/telemarketing.keras",
    "partie3/outputs/models/best_fashion.keras": "deploy/models/best_fashion.keras",
    "partie3/outputs/models/bank-tel.pkl":       "deploy/models/bank-tel.pkl",
}
os.makedirs("deploy/models", exist_ok=True)
ok = 0
for src, dst in SOURCES.items():
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  ✔ {src}")
        ok += 1
    else:
        print(f"  ⚠ MANQUANT : {src}")
print(f"\n{ok}/{len(SOURCES)} fichiers copiés.")
