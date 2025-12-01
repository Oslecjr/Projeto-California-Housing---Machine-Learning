# 🏠 Projeto California Housing - Machine Learning

##Link do COLAB:
https://colab.research.google.com/drive/102zYtBNFtsaUqZbJXucQFVnJTYO7pfC9?usp=sharing

## 📋 Descrição
Projeto acadêmico para previsão de preços de imóveis na Califórnia usando múltiplos algoritmos de Machine Learning, desenvolvido para a disciplina de Programação Avançada.

## 🎯 Objetivo
Comparar o desempenho de diferentes modelos de Machine Learning na tarefa de regressão para prever preços médios de casas na Califórnia.

## 📊 Dataset
- **Fonte**: scikit-learn - `fetch_california_housing`
- **Instâncias**: 20.640
- **Features**: 8
- **Target**: Preço médio das casas (em $100,000)

### Features:
1. `MedInc` - Renda média da região
2. `HouseAge` - Idade média das casas
3. `AveRooms` - Número médio de cômodos
4. `AveBedrms` - Número médio de quartos
5. `Population` - População da região
6. `AveOccup` - Ocupação média
7. `Latitude` - Latitude
8. `Longitude` - Longitude

## 🤖 Modelos Implementados
- 📈 Linear Regression
- 🌳 Random Forest
- 🔍 Support Vector Regression (SVR)
- 📍 K-Nearest Neighbors (KNN)

## 🛠️ Tecnologias
- Python 3.8+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- jupyter

## 🚀 Como Executar

### Opção 1: Google Colab (Recomendada)
1. Acesse o [Google Colab](https://colab.research.google.com/)
2. Faça upload do arquivo `california_housing_project.ipynb`
3. Execute as células sequencialmente

### Opção 2: Localmente
```bash
# Clone o repositório
git clone https://github.com/seu-usuario/Projeto-California-Housing---Machine-Learning.git

# Entre na pasta
cd Projeto-California-Housing---Machine-Learning

# Instale as dependências
pip install -r requirements.txt

# Execute o notebook
jupyter notebook california_housing_project.ipynb

# Ou execute o script
python main.py
