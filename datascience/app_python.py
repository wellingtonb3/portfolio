from flask import Flask, request, jsonify  # Importa o Flask, request e jsonify
from flask_cors import CORS  # Importa o CORS
import joblib  # Importa o joblib
import unicodedata
import re

# Cria a instância do aplicativo Flask
app = Flask(__name__)

# Habilita CORS para todas as rotas
CORS(app)

# 1. Função de Limpeza (Padroniza o texto removendo acentos e pontuação)
def limpar_texto(texto):
    if not isinstance(texto, str):
        return ""
    # Minúsculas e remover acentos
    texto = texto.lower()
    texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('utf-8')
    # Remover pontuação
    texto = re.sub(r'[^\w\s]', '', texto)
    return texto

# Tenta carregar o modelo treinado e o vetorizador
try:
    data = joblib.load('modelo_utlc_apps_sentimento.pkl')
    model = data['model']
    vectorizer = data['vectorizer']
    print("Modelo carregado com sucesso!")

except Exception as e:
    print(f"Erro: Erro ao carregar o modelo: {e}")
    model = None
    vectorizer = None

# Dicionário para mapear as classes de sentimento
LABEL_MAP = {
    "negative": "Negativo",
    "neutral": "Neutro",
    "positive": "Positivo"
}

# Define a rota para a previsão de sentimentos
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Verifica se o modelo foi carregado antes de tentar usar
        if model is None or vectorizer is None:
             return jsonify({"erro": "Modelo de IA não está disponível."}), 503

        dados = request.get_json()
        texto_bruto = dados.get('text') if dados else None

        if not texto_bruto or len(texto_bruto.strip()) < 5:
            return jsonify({"erro": "Texto não fornecido ou muito curto."}), 400

        # Aplicando a limpeza antes da predição 
        texto_processado = limpar_texto(texto_bruto)
        
        # Transforma o texto limpo em vetor numérico
        X = vectorizer.transform([texto_processado])
        
        # Realiza a predição
        prediction_label = model.predict(X)[0]
        proba = float(model.predict_proba(X).max())

        return jsonify({
            "previsao": LABEL_MAP.get(prediction_label, "Desconhecido"),
            "probabilidade": round(proba, 2),
            "texto_processado": texto_processado  # Opcional: para você conferir a limpeza
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"erro": str(e)}), 500

if __name__ == '__main__':
    # Roda o servidor
    app.run(host="0.0.0.0", port=5000)