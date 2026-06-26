"""
================================================================================
 AIoT Smart Irrigation System — Professional Model Training Pipeline
================================================================================
 Dataset  : Smart Agriculture Dataset (Kaggle - Public)
 Model    : Random Forest Classifier with Pipeline Preprocessing
 Date     : Juni 2026
 Version  : 2.0 (High Performance Edition)
================================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_curve, auc, precision_recall_curve
)
import joblib
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# SETTINGS & CONFIGURATIONS (Academic Publication Style)
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')
RANDOM_STATE = 42
TEST_SIZE = 0.2

os.makedirs(PLOTS_DIR, exist_ok=True)

# Styling Matplotlib untuk laporan ilmiah
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.color'] = '#333333'
plt.rcParams['ytick.color'] = '#333333'
plt.rcParams['grid.color'] = '#EEEEEE'
plt.rcParams['grid.linewidth'] = 0.5

LABEL_NAMES = {
    0: 'Tidak Siram',
    1: 'Siram',
    2: 'Siram Banyak'
}


def find_dataset():
    """Mendeteksi file dataset di direktori."""
    for f in os.listdir(DATASET_DIR):
        if f.endswith('.csv'):
            path = os.path.join(DATASET_DIR, f)
            print(f"[STATUS] Dataset terdeteksi: {f}")
            return path
    print("[ERROR] File dataset CSV tidak ditemukan.")
    sys.exit(1)


def load_dataset(csv_path):
    """Membaca dataset dan menampilkan karakteristik data."""
    print("\n" + "="*80)
    print("  TAHAP 1: MEMBACA DAN MEMVERIFIKASI DATASET")
    print("="*80)
    
    df = pd.read_csv(csv_path)
    print(f"Dimensi Data: {df.shape[0]} baris, {df.shape[1]} kolom")
    print("\nStruktur Kolom:")
    for col in df.columns:
        print(f" - {col} ({df[col].dtype})")
    
    return df


def generate_professional_plots(df):
    """
    TAHAP 2: Exploratory Data Analysis (EDA) & Visualisasi Standar Publikasi
    """
    print("\n" + "="*80)
    print("  TAHAP 2: EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*80)
    
    num_features = ['MOI', 'temp', 'humidity']
    target_col = 'result'
    
    # 1. Distribusi Fitur Numerik
    print("Menghasilkan visualisasi: Distribusi Fitur Numerik")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = ['#1f77b4', '#d62728', '#2ca02c']
    feature_labels = ['Moisture Index (MOI)', 'Temperature (°C)', 'Humidity (%)']
    
    for i, (col, color, label) in enumerate(zip(num_features, colors, feature_labels)):
        sns.histplot(df[col], bins=35, ax=axes[i], color=color, kde=True, edgecolor='white', alpha=0.7)
        axes[i].set_title(f'Distribusi {label}', fontsize=12, fontweight='semibold', pad=10)
        axes[i].set_xlabel(label, fontsize=10)
        axes[i].set_ylabel('Frekuensi', fontsize=10)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].grid(axis='y', linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'feature_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Correlation Matrix Heatmap
    print("Menghasilkan visualisasi: Heatmap Korelasi")
    fig, ax = plt.subplots(figsize=(7.5, 6))
    corr = df[num_features + [target_col]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    
    sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, fmt='.3f', square=True,
                linewidths=1.5, cbar_kws={'shrink': 0.8}, mask=mask, ax=ax,
                annot_kws={'size': 11, 'weight': 'semibold'})
    ax.set_title('Matriks Korelasi Fitur Numerik', fontsize=13, fontweight='semibold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Class Balance
    print("Menghasilkan visualisasi: Distribusi Kelas Target")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    counts = df[target_col].value_counts().sort_index()
    labels = [LABEL_NAMES[k] for k in counts.index]
    palette = ['#4ca64c', '#ffaa66', '#d9534f']
    
    bars = ax.bar(labels, counts.values, color=palette, edgecolor='#777777', linewidth=0.8, width=0.5)
    
    # Menambahkan anotasi di atas bar
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}\n({height/len(df)*100:.1f}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9.5, fontweight='semibold')
                    
    ax.set_title('Distribusi Kelas Target Keputusan Irigasi', fontsize=12, fontweight='semibold', pad=15)
    ax.set_ylabel('Frekuensi', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'class_balance.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Boxplot per Kelas
    print("Menghasilkan visualisasi: Boxplot Fitur per Kelas Target")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, (col, label) in enumerate(zip(num_features, feature_labels)):
        sns.boxplot(x=target_col, y=col, data=df, ax=axes[i], palette=palette, width=0.5, linewidth=1.2)
        axes[i].set_xticklabels(labels, fontsize=9.5)
        axes[i].set_title(f'Statistik {label} per Kelas', fontsize=11.5, fontweight='semibold')
        axes[i].set_xlabel('Keputusan Irigasi', fontsize=9.5)
        axes[i].set_ylabel(col, fontsize=9.5)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].grid(axis='y', linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'boxplot_per_class.png'), dpi=300, bbox_inches='tight')
    plt.close()


def build_pipeline(X_train):
    """
    TAHAP 3: Membangun Preprocessing & Model Pipeline
    """
    print("\n" + "="*80)
    print("  TAHAP 3: MEMBANGUN PIPELINE PREPROCESSING & MODEL")
    print("="*80)
    
    num_cols = ['MOI', 'temp', 'humidity']
    cat_cols = ['crop ID', 'soil_type', 'Seedling Stage']
    
    print(f"Fitur Numerik    : {num_cols}")
    print(f"Fitur Kategorikal: {cat_cols}")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=150,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])
    
    return pipeline


def evaluate_and_plot_results(pipeline, X_train, X_test, y_train, y_test):
    """
    TAHAP 4 & 5: Evaluasi Akhir & Visualisasi Model Performance
    """
    print("\n" + "="*80)
    print("  TAHAP 4 & 5: EVALUASI PIPELINE MODEL")
    print("="*80)
    
    # Prediksi
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)
    
    # Metrik Dasar
    acc = accuracy_score(y_test, y_pred)
    precision_w = precision_score(y_test, y_pred, average='weighted')
    recall_w = recall_score(y_test, y_pred, average='weighted')
    f1_w = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Hasil Evaluasi General:")
    print(f" - Akurasi Global     : {acc:.5f} ({acc*100:.3f}%)")
    print(f" - Weighted Precision : {precision_w:.5f}")
    print(f" - Weighted Recall    : {recall_w:.5f}")
    print(f" - Weighted F1-Score  : {f1_w:.5f}")
    
    print("\nLaporan Klasifikasi Detail:")
    target_names = [LABEL_NAMES[i] for i in sorted(y_test.unique())]
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # 1. Plot Confusion Matrix
    print("Menghasilkan visualisasi: Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names,
                linewidths=1.5, linecolor='white', ax=ax, square=True,
                annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_xlabel('Prediksi Model', fontsize=11, fontweight='semibold', labelpad=10)
    ax.set_ylabel('Data Aktual', fontsize=11, fontweight='semibold', labelpad=10)
    ax.set_title(f'Confusion Matrix\nAkurasi: {acc*100:.2f}%', fontsize=13, fontweight='semibold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Plot Feature Importance (dari model Random Forest di dalam Pipeline)
    print("Menghasilkan visualisasi: Feature Importance")
    # Dapatkan nama fitur setelah preprocessing
    preprocessor = pipeline.named_steps['preprocessor']
    
    # Dapatkan nama kolom numerik
    num_cols = ['MOI', 'temp', 'humidity']
    
    # Dapatkan nama kolom one-hot encoded
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_cols_encoded = list(cat_encoder.get_feature_names_out(['crop ID', 'soil_type', 'Seedling Stage']))
    
    feature_names = num_cols + cat_cols_encoded
    importances = pipeline.named_steps['classifier'].feature_importances_
    
    # Sorting
    indices = np.argsort(importances)[::-1]
    
    # Ambil top 15 fitur paling berpengaruh agar plot tidak terlalu padat
    top_n = min(15, len(feature_names))
    top_indices = indices[:top_n]
    
    fig, ax = plt.subplots(figsize=(8.5, 6))
    colors = plt.cm.viridis(np.linspace(0.8, 0.2, top_n))
    
    bars = ax.barh(range(top_n), importances[top_indices][::-1], color=colors[::-1], edgecolor='#555555', linewidth=0.5, height=0.6)
    
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in top_indices][::-1], fontsize=9)
    ax.set_xlabel('Skor Kepentingan (Importance Score)', fontsize=10)
    ax.set_title(f'Top {top_n} Fitur Paling Berpengaruh dalam Prediksi', fontsize=12, fontweight='semibold', pad=15)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for bar in bars:
        width = bar.get_width()
        ax.annotate(f'{width:.4f}',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0),  # 5 points horizontal offset
                    textcoords="offset points",
                    ha='left', va='center', fontsize=8.5, fontweight='semibold')
                    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Simpan objek summary untuk laporan backend
    summary = {
        'accuracy': acc,
        'precision_w': precision_w,
        'recall_w': recall_w,
        'f1_score_w': f1_w,
        'feature_names': feature_names,
        'feature_importances': dict(zip(feature_names, importances.tolist())),
        'model_type': 'Pipeline-RandomForestClassifier',
        'n_estimators': 150,
        'dataset_shape': X_train.shape[0] + X_test.shape[0],
    }
    
    joblib.dump(summary, os.path.join(OUTPUT_DIR, 'training_summary.pkl'))
    print("Summary metrik evaluasi berhasil disimpan.")
    
    return summary


def main():
    print("================================================================================")
    print("  MEMULAI PIPELINE PELATIHAN MODEL AIoT IRIGASI CERDAS")
    print("================================================================================")
    
    # 1. Cari & Load data
    csv_path = find_dataset()
    df = load_dataset(csv_path)
    
    # 2. EDA & Visualisasi
    generate_professional_plots(df)
    
    # 3. Split dataset
    X = df.drop(columns=['result'])
    y = df['result']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # 4. Build Pipeline
    pipeline = build_pipeline(X_train)
    
    # 5. Training
    print("\nMelakukan training model pipeline...")
    pipeline.fit(X_train, y_train)
    print("Training selesai.")
    
    # 6. Evaluasi & Save
    evaluate_and_plot_results(pipeline, X_train, X_test, y_train, y_test)
    
    # Simpan pipeline model secara utuh
    pipeline_path = os.path.join(OUTPUT_DIR, 'model_pipeline.pkl')
    joblib.dump(pipeline, pipeline_path)
    print(f"\n[SUKSES] Pipeline model tersimpan di: {pipeline_path}")
    print("Proses pembuatan model selesai dengan sukses.")


if __name__ == '__main__':
    main()
