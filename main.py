"""
Projeto California Housing - Machine Learning
Script principal para execução via terminal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def main():
    print("🚀 Iniciando Projeto California Housing...")
    
    # Carregar dados
    data = fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target
    
    print(f"Dataset carregado: {X.shape[0]} amostras, {X.shape[1]} features")
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Pré-processamento
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Treinar modelo
    print("Treinando Random Forest...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Avaliar
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"✅ Modelo treinado!")
    print(f"📊 RMSE: {rmse:.4f}")
    print(f"📊 R²: {r2:.4f}")
    
    # Salvar modelo
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/random_forest_model.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    
    print("💾 Modelo salvo na pasta 'models/'")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': data.feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🎯 Feature Importance:")
    print(feature_importance)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Importância das Features - Random Forest')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("📈 Gráfico salvo como 'feature_importance.png'")

if __name__ == "__main__":
    main()
