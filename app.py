import os
from flask import Flask, request, jsonify, render_template
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import openai
from openai import OpenAI
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Inicializar Flask
app = Flask(__name__)

# Cargar el modelo de embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Leer y cargar datos
file_path = "./Data/Entrevista_LuisaGonzales_JuanCueva.csv"
data = pd.read_csv(file_path, sep=",")
documents = data['Entrevista'].tolist()
ids = data['ID'].tolist()
metadatas = data[['Candidato', 'Temas', 'Descripción']].to_dict(orient='records')

# Generar embeddings y almacenar en ChromaDB
embeddings = model.encode(documents).tolist()
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("prueba_collection")

# Agregar documentos si la colección está vacía
if len(collection.get()) == 0:
    collection.add(documents=documents, metadatas=metadatas, ids=ids, embeddings=embeddings)

# Configuración de OpenAI
cliente = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Ruta para la página de inicio
@app.route('/')
def index():
    return render_template('index.html')  # Renderiza el archivo index.html

# Ruta para consultar documentos relevantes y generar una respuesta
@app.route('/query', methods=['POST'])
def query_system():
    try:
        user_query = request.json.get('query')
        if not user_query:
            return jsonify({"error": "No se proporcionó ninguna consulta."}), 400

        # Generar embedding de la consulta
        query_embedding = model.encode([user_query]).tolist()

        # Consultar documentos más similares
        results = collection.query(query_embeddings=query_embedding, n_results=3)

        # Preparar el contexto para el modelo GPT
        context = "\n".join(results['documents'][0]) if results['documents'] else "No se encontraron documentos relevantes."

        # Usar GPT para generar una respuesta basada en el contexto
        response = cliente.chat.completions.create(
            model="gpt-4o-mini",  # Cambiar a un modelo compatible con OpenAI
            messages=[
                {"role": "system", "content": "Eres un experto en recuperación de información que proporciona respuestas breves y concisas."},
                {"role": "user", "content": f"Usando el siguiente contexto, responde de manera breve y concisa a la pregunta: '{user_query}'\n\nContexto:\n{context}\n\nRespuesta:"}
            ]
        )

        generated_response = response.choices[0].message.content
        return jsonify({"query": user_query, "response": generated_response, "context": context})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run(debug=True)