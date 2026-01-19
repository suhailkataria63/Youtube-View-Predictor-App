#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==========================
# macOS bundle runtime shims
# ==========================
import os, sys, pathlib, datetime, traceback
if getattr(sys, "frozen", False):
    # Ensure bundled Qt plugins and dylibs are found
    exe_dir = os.path.dirname(sys.executable)
    fw = os.path.abspath(os.path.join(exe_dir, "..", "Frameworks"))
    plugins = os.path.abspath(os.path.join(exe_dir, "..", "PlugIns"))
    os.environ.setdefault("DYLD_LIBRARY_PATH", fw)
    os.environ.setdefault("QT_PLUGIN_PATH", plugins)

# =================
# Crash log capture
# =================
def _bundle_log_path():
    base = os.path.expanduser("~/Library/Logs/yt_view_predictor")
    pathlib.Path(base).mkdir(parents=True, exist_ok=True)
    return os.path.join(base, "crash.log")

def _excepthook(exc_type, exc, tb):
    try:
        with open(_bundle_log_path(), "a", encoding="utf-8") as f:
            f.write("\n=== {}\n".format(datetime.datetime.now().isoformat()))
            traceback.print_exception(exc_type, exc, tb, file=f)
    finally:
        sys.__excepthook__(exc_type, exc, tb)
sys.excepthook = _excepthook

# ======
# Imports
# ======
import math
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from PySide6.QtGui import QFont
QFont.setFamily(QFont(), "Helvetica Neue")  # or "Arial"


from PySide6.QtCore import (
    Qt, QThread, Signal, QPropertyAnimation, QEasingCurve, QRect, QTimer
)
from PySide6.QtGui import (
    QAction, QIcon, QPalette, QColor, QFont, QFontDatabase
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox,
    QDateTimeEdit, QTabWidget, QMessageBox, QToolBar, QStatusBar, QGridLayout,
    QGroupBox, QScrollArea, QSizePolicy, QProgressBar
)

# Matplotlib (QtAgg canvas)
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# XGBoost – import safe path
from xgboost import XGBRegressor

# ===============================
# Deployment-friendly feature set
# ===============================
FEATURES_NUM = ["likes", "dislikes", "comment_count", "days_since_publish"]
FEATURES_CAT = ["category_id", "comments_disabled", "ratings_disabled", "video_error_or_removed",
                "publish_hour", "publish_weekday"]
TARGET_COL = "views"
DATE_COLS = ("publish_time", "trending_date")

# ===========
# Model paths
# ===========
APP_SUPPORT_DIR = os.path.join(os.path.expanduser("~/Library/Application Support/yt_view_predictor"))
Path(APP_SUPPORT_DIR).mkdir(parents=True, exist_ok=True)
MODEL_PATH = os.path.join(APP_SUPPORT_DIR, "yt_xgb_model.pkl")

# =====================
# Feature prep utilities
# =====================
def _parse_dt(s):
    if pd.isna(s):
        return pd.NaT
    try:
        return pd.to_datetime(s, utc=True, errors="coerce")
    except Exception:
        return pd.NaT

def prepare_features(df_in: pd.DataFrame, *, for_train: bool = True):
    import pandas as pd

    def _parse_dt_utc_any(x):
        # Always return tz-aware UTC Timestamp or NaT
        return pd.to_datetime(x, errors="coerce", utc=True)

    df = df_in.copy()

    # Ensure columns exist
    for c in ["likes", "dislikes", "comment_count", "category_id",
              "comments_disabled", "ratings_disabled", "video_error_or_removed",
              "publish_time", "trending_date"]:
        if c not in df.columns:
            df[c] = pd.NA

    # Dtypes
    for b in ["comments_disabled", "ratings_disabled", "video_error_or_removed"]:
        df[b] = df[b].astype("boolean")
    for n in ["likes", "dislikes", "comment_count", "category_id"]:
        df[n] = pd.to_numeric(df[n], errors="coerce")

    # publish_time → UTC-aware
    pub = df["publish_time"].apply(_parse_dt_utc_any)

    # trending_date: try explicit yy.dd.mm (USvideos) else general parse
    trn = pd.NaT
    if "trending_date" in df.columns:
        td = df["trending_date"]
        if isinstance(td, pd.Series) and td.dtype == object and td.astype(str).str.contains(r"^\d{2}\.\d{2}\.\d{2}$").any():
            trn = pd.to_datetime(td, format="%y.%d.%m", errors="coerce", utc=True)
        else:
            trn = _parse_dt_utc_any(td)

    # Reference: prefer trending_date else "now" UTC
    now_utc = pd.Timestamp.now(tz="UTC")
    if isinstance(trn, pd.Series):
        ref = trn.fillna(now_utc)
    else:
        ref = pd.Series([now_utc] * len(df), index=df.index)

    # Engineered features
    df["publish_hour"] = pub.dt.hour.astype("Int64")
    df["publish_weekday"] = pub.dt.weekday.astype("Int64")
    df["days_since_publish"] = ((ref - pub).dt.total_seconds() / 86400.0).astype("float")

    X = df[["likes", "dislikes", "comment_count", "days_since_publish",
            "category_id", "comments_disabled", "ratings_disabled",
            "video_error_or_removed", "publish_hour", "publish_weekday"]].copy()

    y = None
    if for_train and "views" in df.columns:
        y = pd.to_numeric(df["views"], errors="coerce")

    return X, y




# ==================
# Preprocessor build
# ==================
def make_preprocessor():
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer

    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler(with_mean=False))
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh",  OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, FEATURES_NUM),
            ("cat", cat_pipe, FEATURES_CAT)
        ],
        sparse_threshold=1.0
    )
    return pre

# ==============
# Trainer (sync)
# ==============
def tune_and_train(csv_path, save_to=None, *, max_train_rows=20000):
    """
    Single-process trainer compatible with older XGBoost builds (no callbacks).
    - Compact feature set
    - Imputers + OneHotEncoder(sparse_output=True)
    - Early stopping via 'early_stopping_rounds' if supported
    - Saves pipeline to MODEL_PATH (or 'save_to')
    Returns: (mae, r2, y_valid, preds, X_valid_df, y_valid_series, meta)
    """
    import time
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_absolute_error, r2_score

    t0 = time.time()
    df = pd.read_csv(csv_path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Dataset must contain '{TARGET_COL}' as target.")

    X_all, y_all = prepare_features(df, for_train=True)

    if max_train_rows and len(X_all) > max_train_rows:
        idx = X_all.sample(max_train_rows, random_state=42).index
        X_all = X_all.loc[idx]
        y_all = y_all.loc[idx]

    Xtr, Xva, ytr, yva = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

    pre = make_preprocessor()
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=600,
        max_depth=6,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=1.0,
        n_jobs=max(1, (os.cpu_count() or 2) - 1),
        tree_method="hist",
        random_state=42,
        verbosity=0
    )
    try:
        model.set_params(eval_metric="rmse")
    except Exception:
        pass

    # Fit with legacy early stopping if available
    try:
        Xtr_t = pre.fit_transform(Xtr, ytr)
        Xva_t = pre.transform(Xva)
        model.fit(Xtr_t, ytr, eval_set=[(Xva_t, yva)], early_stopping_rounds=50, verbose=False)
    except TypeError:
        # Fallback: no early stopping available
        pipe_tmp = Pipeline([("pre", pre), ("model", model)])
        pipe_tmp.fit(Xtr, ytr)
        pre = pipe_tmp.named_steps["pre"]
        model = pipe_tmp.named_steps["model"]
        Xtr_t = pre.transform(Xtr)
        Xva_t = pre.transform(Xva)

    preds = model.predict(Xva_t)
    from sklearn.metrics import mean_absolute_error, r2_score
    mae = float(mean_absolute_error(yva, preds))
    r2  = float(r2_score(yva, preds))

    pipe = Pipeline([("pre", pre), ("model", model)])

    meta = {
        "mae": mae,
        "r2": r2,
        "src": os.path.basename(csv_path),
        "features_num": FEATURES_NUM,
        "features_cat": FEATURES_CAT,
        "train_rows": int(len(X_all)),
        "train_seconds": round(time.time()-t0, 2)
    }

    out = save_to or os.path.join(os.getcwd(), "yt_xgb_model.pkl")
    joblib.dump({"pipeline": pipe, "meta": meta}, out)

    # Best-effort SHAP on transformed data sample
    shap_meta = {"ok": False}
    try:
        import shap
        # sample
        n = Xva_t.shape[0]
        take = min(400, n)
        if take > 0:
            idx = np.random.RandomState(42).choice(n, take, replace=False)
            Xs = Xva_t[idx]
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(Xs)
            shap_meta = {"ok": True, "sample": int(take), "values_shape": getattr(shap_vals, "shape", None)}
    except Exception as e:
        shap_meta = {"ok": False, "error": str(e)}
    meta["shap"] = shap_meta

    return mae, r2, yva, preds, Xva, yva, meta

# =======================
# Matplotlib canvas class
# =======================
class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, width=5, height=3, dpi=120):
        self.figure = Figure(figsize=(width, height), dpi=dpi, layout="tight")
        super().__init__(self.figure)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(180)

# =====================
# Worker Thread (Train)
# =====================
class TrainWorker(QThread):
    finished = Signal(object)   # payload: dict with metrics/meta or exception string
    def __init__(self, csv_path: str, parent=None):
        super().__init__(parent)
        self.csv_path = csv_path

    def run(self):
        try:
            res = tune_and_train(self.csv_path, save_to=MODEL_PATH)
            payload = {
                "ok": True,
                "mae": res[0],
                "r2": res[1],
                "y_valid": res[2],
                "preds": res[3],
                "X_valid": res[4],
                "meta":  res[6]
            }
        except Exception as e:
            payload = {"ok": False, "error": str(e)}
        self.finished.emit(payload)

# ==================
# Pretty UI helpers
# ==================
# ---- add this helper somewhere near your other helpers ----
def _ax(canvas):
    """Return a fresh, styled Axes for the given FigureCanvas."""
    fig = canvas.figure
    fig.clear()
    ax = fig.add_subplot(111)
    ax.set_facecolor("#1E2126"); fig.set_facecolor("#1E2126")
    for s in ax.spines.values(): s.set_color("#3A3F47")
    ax.tick_params(colors="#C9D1D9")
    return ax

def _to_dense_small(X, cap=400):
    """Take a small random slice of X and ensure it's dense (for SHAP)."""
    import numpy as np
    from scipy import sparse
    n = X.shape[0]
    if n == 0:
        return None, None
    take = min(cap, n)
    idx = np.random.RandomState(42).choice(n, take, replace=False)
    Xs = X[idx]
    if sparse.issparse(Xs):
        Xs = Xs.toarray()
    return Xs, idx


def apply_dark(app: QApplication):
    pal = QPalette()
    # base shades
    pal.setColor(QPalette.Window, QColor(22, 23, 26))
    pal.setColor(QPalette.WindowText, Qt.white)
    pal.setColor(QPalette.Base, QColor(28, 29, 33))
    pal.setColor(QPalette.AlternateBase, QColor(33, 35, 39))
    pal.setColor(QPalette.ToolTipBase, QColor(40, 41, 46))
    pal.setColor(QPalette.ToolTipText, Qt.white)
    pal.setColor(QPalette.Text, Qt.white)
    pal.setColor(QPalette.Button, QColor(33, 35, 39))
    pal.setColor(QPalette.ButtonText, Qt.white)
    pal.setColor(QPalette.BrightText, Qt.red)
    pal.setColor(QPalette.Highlight, QColor(64, 98, 190))
    pal.setColor(QPalette.HighlightedText, Qt.white)
    app.setPalette(pal)
    app.setStyle("fusion")

def fade_in(widget: QWidget, duration=500):
    widget.setWindowOpacity(0.0)
    anim = QPropertyAnimation(widget, b"windowOpacity", widget)
    anim.setDuration(duration)
    anim.setStartValue(0.0)
    anim.setEndValue(1.0)
    anim.setEasingCurve(QEasingCurve.InOutQuad)
    anim.start()
    # Keep a reference
    widget._fade_anim = anim

def card(title: str, big_text: str = "-", subtitle: str = "") -> QFrame:
    box = QFrame()
    box.setObjectName("card")
    box.setStyleSheet("""
        QFrame#card {
            background-color: #1E2126;
            border-radius: 14px;
            border: 1px solid #2A2E35;
        }
        QLabel[role="title"] {
            color: #AAB2BF; font-size: 12px; letter-spacing: 0.8px;
        }
        QLabel[role="value"] {
            color: #FFFFFF; font-size: 24px; font-weight: 600;
        }
        QLabel[role="subtitle"] {
            color: #8B93A1; font-size: 11px;
        }
    """)
    v = QVBoxLayout(box); v.setContentsMargins(14, 12, 14, 12)
    lt = QLabel(title); lt.setProperty("role","title")
    lv = QLabel(big_text); lv.setProperty("role","value")
    ls = QLabel(subtitle); ls.setProperty("role","subtitle")
    v.addWidget(lt); v.addWidget(lv); v.addWidget(ls)
    return box

def spacer(h=8):
    s = QWidget(); s.setFixedHeight(h); return s

# ============
# Main Window
# ============
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YouTube View Predictor")
        self.setWindowIcon(QIcon.fromTheme("applications-graphics"))
        self.setMinimumSize(1100, 720)

        self.pipe = None   # loaded pipeline
        self.meta = {}

        self._build_ui()
        fade_in(self, 600)

        # Try autoload model
        if os.path.exists(MODEL_PATH):
            try:
                obj = joblib.load(MODEL_PATH)
                self.pipe = obj["pipeline"]
                self.meta = obj.get("meta", {})
                self._update_metric_cards()
                self.status.showMessage("Loaded existing model.", 5000)
            except Exception as e:
                self.status.showMessage(f"Model load failed: {e}", 7000)

    # ---- UI build ----
    def _build_ui(self):
        # Toolbar
        tb = QToolBar("Main"); tb.setMovable(False)
        act_train = QAction("Train from CSV", self)
        act_train.triggered.connect(self.on_train_clicked)
        act_save  = QAction("Save Model As…", self)
        act_save.triggered.connect(self.on_save_model)
        tb.addAction(act_train); tb.addAction(act_save)
        self.addToolBar(tb)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)

        # Tabs
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.setTabPosition(QTabWidget.North)
        tabs.setStyleSheet("QTabWidget::pane{border:0;} QTabBar::tab{padding:10px 18px;}")
        self.setCentralWidget(tabs)

        # ---- Dashboard Tab ----
        self.tab_dash = QWidget(); tabs.addTab(self.tab_dash, "Dashboard")
        self._build_dashboard(self.tab_dash)

        # ---- Predict Tab ----
        self.tab_pred = QWidget(); tabs.addTab(self.tab_pred, "Predict")
        self._build_predict(self.tab_pred)

        # ---- Insights Tab ----
        self.tab_ins = QWidget(); tabs.addTab(self.tab_ins, "Insights")
        self._build_insights(self.tab_ins)

    def _build_dashboard(self, parent: QWidget):
        root = QVBoxLayout(parent); root.setContentsMargins(16, 12, 16, 12)
        # Metric cards row
        row = QHBoxLayout(); row.setSpacing(12)
        self.card_mae = card("MEAN ABS ERROR", "-")
        self.card_r2  = card("R² SCORE", "-")
        self.card_src = card("SOURCE", "-", "last training csv")
        row.addWidget(self.card_mae); row.addWidget(self.card_r2); row.addWidget(self.card_src)
        root.addLayout(row)
        root.addWidget(spacer(10))

        # Progress
        self.prog = QProgressBar(); self.prog.setRange(0,0); self.prog.setVisible(False)
        self.prog.setTextVisible(True)
        root.addWidget(self.prog)

        # Charts row
        charts = QHBoxLayout(); charts.setSpacing(12)
        # Feature importance canvas
        self.canvas_imp = MplCanvas(width=5.6, height=3.0, dpi=120)
        box_imp = QFrame(); box_imp.setObjectName("card"); box_imp.setStyleSheet(self.card_mae.styleSheet())
        vl = QVBoxLayout(box_imp); vl.setContentsMargins(12, 10, 12, 10)
        ttl = QLabel("Feature Importance"); ttl.setProperty("role","title")
        vl.addWidget(ttl); vl.addWidget(self.canvas_imp)
        charts.addWidget(box_imp)

        # Residuals canvas
        self.canvas_res = MplCanvas(width=5.6, height=3.0, dpi=120)
        box_res = QFrame(); box_res.setObjectName("card"); box_res.setStyleSheet(self.card_mae.styleSheet())
        vl2 = QVBoxLayout(box_res); vl2.setContentsMargins(12, 10, 12, 10)
        ttl2 = QLabel("Validation Residuals"); ttl2.setProperty("role","title")
        vl2.addWidget(ttl2); vl2.addWidget(self.canvas_res)
        charts.addWidget(box_res)

        root.addLayout(charts)
        root.addStretch(1)

    def _build_predict(self, parent: QWidget):
        root = QVBoxLayout(parent); root.setContentsMargins(16, 12, 16, 12)

        form = QGridLayout(); form.setHorizontalSpacing(14); form.setVerticalSpacing(10)

        self.in_likes = QSpinBox(); self.in_likes.setRange(0, 1_000_000_000)
        self.in_dislikes = QSpinBox(); self.in_dislikes.setRange(0, 1_000_000_000)
        self.in_comments = QSpinBox(); self.in_comments.setRange(0, 1_000_000_000)
        self.in_category = QSpinBox(); self.in_category.setRange(1, 1000)
        self.in_pub = QDateTimeEdit(); self.in_pub.setCalendarPopup(True)
        self.in_pub.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        # Default to now
        from datetime import datetime, timezone
        self.in_pub.setDateTime(datetime.now(timezone.utc).astimezone().replace(microsecond=0))

        self.chk_com_dis = QCheckBox("Comments disabled")
        self.chk_rat_dis = QCheckBox("Ratings disabled")
        self.chk_err_rem = QCheckBox("Video error/removed")

        row = 0
        form.addWidget(QLabel("Likes"), row,0);      form.addWidget(self.in_likes, row,1);    row+=1
        form.addWidget(QLabel("Dislikes"), row,0);   form.addWidget(self.in_dislikes, row,1); row+=1
        form.addWidget(QLabel("Comment count"), row,0); form.addWidget(self.in_comments, row,1); row+=1
        form.addWidget(QLabel("Category ID"), row,0); form.addWidget(self.in_category, row,1); row+=1
        form.addWidget(QLabel("Publish time (local)"), row,0); form.addWidget(self.in_pub, row,1); row+=1
        form.addWidget(self.chk_com_dis, row,0); form.addWidget(self.chk_rat_dis, row,1); row+=1
        form.addWidget(self.chk_err_rem, row,0); row+=1

        # Action row
        act_row = QHBoxLayout()
        btn_pred = QPushButton("Predict Views"); btn_pred.clicked.connect(self.on_predict_clicked)
        self.out_pred = card("PREDICTED VIEWS", "-", "point estimate")
        act_row.addWidget(btn_pred); act_row.addStretch(1); act_row.addWidget(self.out_pred)

        root.addLayout(form)
        root.addWidget(spacer(6))
        root.addLayout(act_row)
        root.addStretch(1)

    def _build_insights(self, parent: QWidget):
        root = QVBoxLayout(parent); root.setContentsMargins(16,12,16,12)

        # SHAP canvas
        self.canvas_shap = MplCanvas(width=11.0, height=3.6, dpi=110)
        box = QFrame(); box.setObjectName("card"); box.setStyleSheet(self.card_mae.styleSheet())
        vl = QVBoxLayout(box); vl.setContentsMargins(12,10,12,10)
        ttl = QLabel("SHAP Summary (best-effort)"); ttl.setProperty("role","title")
        vl.addWidget(ttl); vl.addWidget(self.canvas_shap)

        root.addWidget(box)
        root.addStretch(1)

    # -------------
    # UI Interactions
    # -------------
    def on_train_clicked(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Training CSV", str(Path.home()), "CSV Files (*.csv)")
        if not path:
            return
        self.status.showMessage("Training… this may take a moment.")
        self.prog.setVisible(True)
        QApplication.setOverrideCursor(Qt.BusyCursor)

        self.worker = TrainWorker(path, self)
        self.worker.finished.connect(self._on_trained)
        self.worker.start()

    def _on_trained(self, payload: dict):
        self.prog.setVisible(False)
        QApplication.restoreOverrideCursor()
        if not payload.get("ok"):
            QMessageBox.critical(self, "Training failed", str(payload.get("error")))
            self.status.showMessage("Training failed.", 5000)
            return
        try:
            obj = joblib.load(MODEL_PATH)
            self.pipe = obj["pipeline"]
            self.meta = obj.get("meta", {})
        except Exception as e:
            QMessageBox.warning(self, "Model load", f"Model saved but failed to load: {e}")

        self._update_metric_cards()
        self._draw_feature_importance()
        self._draw_residuals(payload.get("y_valid"), payload.get("preds"))
        self._draw_shap(payload.get("X_valid"))   # <-- use validation DF from payload
        self.status.showMessage("Training complete.", 5000)

    def _update_metric_cards(self):
        mae = self.meta.get("mae", None)
        r2  = self.meta.get("r2", None)
        src = self.meta.get("src", "-")
        self._set_card_value(self.card_mae, f"{mae:,.0f}" if mae is not None else "-")
        self._set_card_value(self.card_r2,  f"{r2:.3f}" if r2 is not None else "-")
        self._set_card_subtitle(self.card_src, src)

    def _set_card_value(self, card_widget: QFrame, text: str):
        for i in range(card_widget.layout().count()):
            w = card_widget.layout().itemAt(i).widget()
            if isinstance(w, QLabel) and w.property("role") == "value":
                w.setText(text); break

    def _set_card_subtitle(self, card_widget: QFrame, text: str):
        for i in range(card_widget.layout().count()):
            w = card_widget.layout().itemAt(i).widget()
            if isinstance(w, QLabel) and w.property("role") == "subtitle":
                w.setText(text); break

    def _draw_feature_importance(self):
        if not self.pipe:
            return
        try:
            model = self.pipe.named_steps["model"]
            pre   = self.pipe.named_steps["pre"]
        except Exception:
            return

        try:
            names = pre.get_feature_names_out()
        except Exception:
            names = np.array(FEATURES_NUM + FEATURES_CAT)

        fi = getattr(model, "feature_importances_", None)
        ax = _ax(self.canvas_imp)

        if fi is None or len(fi) != len(names):
            ax.text(0.5, 0.5, "Feature importances unavailable",
                    color="w", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off(); self.canvas_imp.draw(); return

        order = np.argsort(fi)[::-1][:15]
        ax.barh(range(len(order)), fi[order][::-1])
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(names[order][::-1], fontsize=9, color="#C9D1D9")
        ax.set_xlabel("Importance", color="#C9D1D9")
        ax.set_title("Top Features", color="w", pad=8)
        self.canvas_imp.draw()



    def _draw_residuals(self, y_true, y_pred):
        if y_true is None or y_pred is None:
            return
        ax = _ax(self.canvas_res)
        res = np.asarray(y_true) - np.asarray(y_pred)
        ax.scatter(y_pred, res, s=10, alpha=0.5)
        ax.axhline(0, color="#888888", linewidth=1.0)
        ax.set_xlabel("Predicted", color="#C9D1D9")
        ax.set_ylabel("Residuals (y - ŷ)", color="#C9D1D9")
        ax.set_title("Validation Residuals", color="w", pad=8)
        self.canvas_res.draw()



    def _draw_shap(self, X_valid_df):
        """
        Compute SHAP values on a small, dense slice of the transformed validation set,
        then plot a robust bar chart of mean |SHAP| per transformed feature.
        """
        ax = _ax(self.canvas_shap)
        if not self.pipe or X_valid_df is None or len(X_valid_df) == 0:
            ax.text(0.5, 0.5, "SHAP not available (no validation data).",
                    color="w", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off(); self.canvas_shap.draw(); return

        try:
            pre = self.pipe.named_steps["pre"]
            model = self.pipe.named_steps["model"]
        except Exception:
            ax.text(0.5, 0.5, "SHAP not available (missing pre/model).",
                    color="w", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off(); self.canvas_shap.draw(); return

        # Transform validation to numeric space and take a small dense slice
        Xva_t = pre.transform(X_valid_df)
        Xs, _ = _to_dense_small(Xva_t, cap=400)
        if Xs is None:
            ax.text(0.5, 0.5, "SHAP not available (empty validation).",
                    color="w", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off(); self.canvas_shap.draw(); return

        # Transformed feature names
        try:
            names = pre.get_feature_names_out()
        except Exception:
            names = np.array(FEATURES_NUM + FEATURES_CAT)

        try:
            import shap, numpy as np
            explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
            shap_vals = explainer.shap_values(Xs)
            shap_vals = np.asarray(shap_vals, dtype=float)

            mean_abs = np.mean(np.abs(shap_vals), axis=0)
            order = np.argsort(mean_abs)[::-1][:20]

            ax.barh(range(len(order)), mean_abs[order][::-1])
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels(names[order][::-1], fontsize=9, color="#C9D1D9")
            ax.set_xlabel("mean |SHAP| (model space)", color="#C9D1D9")
            ax.set_title("SHAP Feature Impact (Top 20)", color="w", pad=8)
            self.canvas_shap.draw()
        except Exception as e:
            ax.text(0.5, 0.5, f"SHAP failed:\n{e}",
                    color="w", ha="center", va="center", transform=ax.transAxes, wrap=True)
            ax.set_axis_off(); self.canvas_shap.draw()




    def on_save_model(self):
        if not self.pipe:
            QMessageBox.information(self, "Save Model", "Train a model first.")
            return
        p, _ = QFileDialog.getSaveFileName(self, "Save Model", str(Path.home() / "yt_xgb_model.pkl"), "Pickle (*.pkl)")
        if not p:
            return
        try:
            joblib.dump({"pipeline": self.pipe, "meta": self.meta}, p)
            self.status.showMessage(f"Saved model to {p}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))

    def on_predict_clicked(self):
        if not self.pipe:
            QMessageBox.information(self, "Predict", "Please train a model first.")
            return
        # Build single-row input
        from datetime import timezone
        pub_local = self.in_pub.dateTime().toPython()  # timezone-aware? Qt returns naive local
        # Convert to ISO string local; prepare_features is robust
        pub_str = pub_local.isoformat(sep=" ", timespec="seconds")

        row = {
            "likes": int(self.in_likes.value()),
            "dislikes": int(self.in_dislikes.value()),
            "comment_count": int(self.in_comments.value()),
            "category_id": int(self.in_category.value()),
            "comments_disabled": bool(self.chk_com_dis.isChecked()),
            "ratings_disabled": bool(self.chk_rat_dis.isChecked()),
            "video_error_or_removed": bool(self.chk_err_rem.isChecked()),
            "publish_time": pub_str,
            "trending_date": pd.NaT
        }
        X_raw = pd.DataFrame([row])
        X_pred, _ = prepare_features(X_raw, for_train=False)

        try:
            y_hat = self.pipe.predict(X_pred)[0]
            self._set_card_value(self.out_pred, f"{y_hat:,.0f}")
            self.status.showMessage("Prediction complete.", 4000)
            self._pulse(self.out_pred)  # subtle animation
        except Exception as e:
            QMessageBox.critical(self, "Prediction failed", str(e))

    def _pulse(self, widget: QWidget):
        # subtle grow/shrink animation on prediction card
        g = widget.geometry()
        anim = QPropertyAnimation(widget, b"geometry", widget)
        anim.setDuration(300)
        anim.setStartValue(g)
        anim.setKeyValueAt(0.5, QRect(g.x()-6, g.y()-4, g.width()+12, g.height()+8))
        anim.setEndValue(g)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.start()
        widget._pulse_anim = anim

# =====
# main
# =====
def main():
    app = QApplication(sys.argv)

    # Dark theme
    apply_dark(app)

    # Nice default font (fallbacks handled by macOS)
    f = QFont("SF Pro Text", 11)
    app.setFont(f)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
