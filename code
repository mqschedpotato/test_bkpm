import streamlit as st
import os
from pinecone import Pinecone, Index, PodSpec
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings # Untuk kemudahan integrasi dengan Langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader
from dotenv import load_dotenv

# Memuat variabel lingkungan dari .env file
load_dotenv()

# --- Konfigurasi Streamlit ---
st.set_page_config(page_title="Chatbot Tanya Jawab dengan Pinecone & OpenAI", layout="wide")
st.title("🤖 Chatbot Tanya Jawab Cerdas")
st.markdown("""
    Unggah dokumen Anda, biarkan saya mengindeksnya, dan kemudian ajukan pertanyaan!
    Saya akan mencari informasi relevan dan menjawab menggunakan kekuatan OpenAI GPT.
""")

# --- Sidebar untuk API Keys ---
with st.sidebar:
    st.header("Konfigurasi API Keys")
    openai_api_key = st.text_input("OpenAI API Key", type="password", help="Dapatkan dari platform.openai.com")
    pinecone_api_key = st.text_input("Pinecone API Key", type="password", help="Dapatkan dari app.pinecone.io")
    pinecone_environment = st.text_input("Pinecone Environment (Region)", value="gcp-starter", help="Contoh: gcp-starter, us-west-2") # Sesuaikan dengan environment Pinecone Anda

    if openai_api_key and pinecone_api_key and pinecone_environment:
        st.success("API Keys berhasil dikonfigurasi!")
    else:
        st.warning("Mohon masukkan semua API Keys.")

# --- Inisialisasi Klien ---
# Fungsi untuk inisialisasi klien agar tidak diinisialisasi ulang setiap refresh
@st.cache_resource
def get_openai_client(api_key):
    if not api_key:
        st.error("OpenAI API Key belum dimasukkan.")
        return None
    return OpenAI(api_key=api_key)

@st.cache_resource
def get_pinecone_client(api_key, environment):
    if not api_key or not environment:
        st.error("Pinecone API Key atau Environment belum dimasukkan.")
        return None
    try:
        pc = Pinecone(api_key=api_key, environment=environment)
        return pc
    except Exception as e:
        st.error(f"Gagal menginisialisasi Pinecone: {e}")
        return None

@st.cache_resource
def get_openai_embeddings(api_key):
    if not api_key:
        st.error("OpenAI API Key belum dimasukkan.")
        return None
    return OpenAIEmbeddings(openai_api_key=api_key, model="text-embedding-ada-002")

# --- Variabel Global (untuk menyimpan indeks Pinecone setelah dibuat) ---
if "pinecone_index" not in st.session_state:
    st.session_state.pinecone_index = None

# --- Fungsi untuk Memproses Dokumen dan Mengindeks ke Pinecone ---
def process_document_and_index(uploaded_file, pinecone_client, openai_embedder, index_name="my-qa-index"):
    if not uploaded_file:
        st.error("Tidak ada file yang diunggah.")
        return

    # Simpan file sementara
    file_path = os.path.join("./temp", uploaded_file.name)
    os.makedirs("./temp", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Muat dokumen berdasarkan ekstensi
    loader = None
    if uploaded_file.name.endswith(".txt"):
        loader = TextLoader(file_path)
    elif uploaded_file.name.endswith(".md"):
        loader = UnstructuredMarkdownLoader(file_path)
    else:
        st.error("Jenis file tidak didukung. Mohon unggah file .txt atau .md.")
        os.remove(file_path)
        return

    documents = loader.load()

    # Bagi dokumen menjadi chunk
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    st.info(f"Berhasil membagi dokumen menjadi {len(chunks)} bagian (chunks).")

    # Siapkan data untuk diunggah ke Pinecone
    vectors_to_upsert = []
    for i, chunk in enumerate(chunks):
        # Embed chunk menggunakan OpenAI Embeddings
        # openai_embedder.embed_query mengembalikan list embedding, kita ambil yang pertama
        embedding = openai_embedder.embed_query(chunk.page_content)
        vectors_to_upsert.append({
            "id": f"doc-{i}-{hash(uploaded_file.name)}", # ID unik untuk setiap chunk
            "values": embedding,
            "metadata": {"text": chunk.page_content, "source": uploaded_file.name}
        })

    # Cek atau buat indeks Pinecone
    if index_name not in pinecone_client.list_indexes():
        st.info(f"Membuat indeks Pinecone '{index_name}'...")
        pinecone_client.create_index(
            name=index_name,
            dimension=1536, # Dimensi embedding ada-002
            metric='cosine',
            spec=PodSpec(environment=st.session_state.pinecone_environment)
        )
        st.success(f"Indeks '{index_name}' berhasil dibuat.")

    pinecone_index = pinecone_client.Index(index_name)
    st.session_state.pinecone_index = pinecone_index # Simpan indeks di session state

    # Upsert vectors ke Pinecone
    st.info("Mengunggah embedding ke Pinecone...")
    pinecone_index.upsert(vectors=vectors_to_upsert)
    st.success(f"{len(vectors_to_upsert)} embedding berhasil diunggah ke Pinecone!")

    # Hapus file sementara
    os.remove(file_path)

# --- Bagian Unggah File ---
st.header("Unggah Dokumen Anda")
uploaded_file = st.file_uploader("Pilih file .txt atau .md", type=["txt", "md"])

if uploaded_file and st.button("Proses dan Indeks Dokumen"):
    if openai_api_key and pinecone_api_key and pinecone_environment:
        openai_client_obj = get_openai_client(openai_api_key)
        pinecone_client_obj = get_pinecone_client(pinecone_api_key, pinecone_environment)
        openai_embedder_obj = get_openai_embeddings(openai_api_key)

        if openai_client_obj and pinecone_client_obj and openai_embedder_obj:
            with st.spinner("Memproses dan mengindeks dokumen..."):
                process_document_and_index(uploaded_file, pinecone_client_obj, openai_embedder_obj)
        else:
            st.error("Gagal menginisialisasi klien. Periksa API Keys Anda.")
    else:
        st.warning("Mohon masukkan semua API Keys di sidebar.")

# --- Bagian Chatbot ---
st.header("Ajukan Pertanyaan Anda")
user_query = st.text_input("Ketik pertanyaan Anda di sini:")

if st.button("Tanyakan"):
    if not user_query:
        st.warning("Mohon ketikkan pertanyaan Anda.")
    elif st.session_state.pinecone_index is None:
        st.warning("Mohon unggah dan indeks dokumen terlebih dahulu.")
    elif not openai_api_key:
        st.warning("Mohon masukkan OpenAI API Key di sidebar.")
    else:
        openai_client_obj = get_openai_client(openai_api_key)
        openai_embedder_obj = get_openai_embeddings(openai_api_key)

        if openai_client_obj and openai_embedder_obj:
            with st.spinner("Mencari jawaban..."):
                # 1. Embed query pengguna
                query_embedding = openai_embedder_obj.embed_query(user_query)

                # 2. Cari dokumen relevan di Pinecone
                search_results = st.session_state.pinecone_index.query(
                    vector=query_embedding,
                    top_k=5, # Ambil 5 dokumen teratas
                    include_metadata=True
                )

                # Gabungkan teks dari hasil pencarian
                context = ""
                for res in search_results.matches:
                    context += res.metadata['text'] + "\n\n"

                if not context:
                    st.info("Maaf, saya tidak menemukan informasi relevan di dokumen yang Anda berikan.")
                else:
                    # 3. Buat prompt untuk GPT
                    prompt = f"""
                    Anda adalah asisten AI yang membantu menjawab pertanyaan berdasarkan konteks yang diberikan.
                    Gunakan informasi berikut untuk menjawab pertanyaan pengguna. Jika jawabannya tidak ada dalam konteks,
                    katakan bahwa Anda tidak memiliki informasi tersebut.

                    Konteks:
                    {context}

                    Pertanyaan: {user_query}
                    Jawaban:
                    """

                    # 4. Dapatkan jawaban dari OpenAI GPT
                    try:
                        response = openai_client_obj.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "Anda adalah asisten yang membantu menjawab pertanyaan berdasarkan konteks yang diberikan."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.7, # Kreativitas jawaban
                            max_tokens=500 # Batas panjang jawaban
                        )
                        st.success("Jawaban:")
                        st.write(response.choices[0].message.content)
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat meminta jawaban dari OpenAI: {e}")
        else:
            st.error("Gagal menginisialisasi klien OpenAI. Periksa API Key Anda.")
