import os
import gdown
import tempfile
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Map category -> list of Google Drive direct download links
DOC_CATEGORIES = {
    "Communications": [
        "https://drive.google.com/uc?id=19L3X2FPUsQ0ZjRD5I2BanUqpwj2xOcB_",
        "https://drive.google.com/uc?id=1dcc1nsjYWZuSM-reffs3zDfGu6D9QKMZ",
        "https://drive.google.com/uc?id=1rWy6UNZnYjJmIho7Qm-6aqdWbjV-enAo",
        "https://drive.google.com/uc?id=1sWrZBRjDeruD3X-Qz-3NaIuQCro0yluW",
        "https://drive.google.com/uc?id=1klJUyGG65AohFAAAji5-q4A34vUSphSn",
        "https://drive.google.com/uc?id=18s6vyAQSqR4O9h4IpjrxLZRxn0B09Z-X",
        "https://drive.google.com/uc?id=1MzKi05bykV1Kn8EXgJ7HtYaTpZ0jUmBN",
        "https://drive.google.com/uc?id=1V_GP2B2Nc2FbKQROZHH4x_nVfCVnxBiV",
        "https://drive.google.com/uc?id=1ZJZQUqk2P1moz1o-cM03N3rmHJNIUb01",
        "https://drive.google.com/uc?id=1S5lzSxnfIy4NIV3l2wHkbQGAqQK_P7mh",
        "https://drive.google.com/uc?id=1IYfcfdcE3t4UevVVvffqUv5sTmkfGhH8",
        "https://drive.google.com/uc?id=1TyTnXp1lcfyggAO3GLC90co2LM1GRlwa",
        "https://drive.google.com/uc?id=1I3QUiO18V5ubqdEMeNYiXfUVueUscKeI",
        "https://drive.google.com/uc?id=1oCT5Z6vUEdoEtxzPyuNDIiBt-PG-nnFV",
        "https://drive.google.com/uc?id=1_uaSuhvFaiE987ukpeFWriKWa2YK9Kkq",
        "https://drive.google.com/uc?id=13FbMkq5Kald81Hdmsvzbn-OhBQw4nO6e",
        "https://drive.google.com/uc?id=1HuVzkEhM-ld8GyxszvZZzcdjCraZad_u",
        "https://drive.google.com/uc?id=1_jbuHvAuIYzZNdhdDZlJlmajDwB2ge7S",
        "https://drive.google.com/uc?id=1UCuQmMwrysIEyrCsx0uikSsvcjdgagvd",
        "https://drive.google.com/uc?id=12f9nLrnfD62walsTZxRF4YADUH1rNTVw",
        "https://drive.google.com/uc?id=1OGfumIZfleoL29enakm4q_xg1OeFXFZL",
        "https://drive.google.com/uc?id=19tO9vcGALvnx48Dq9r4iildxPnH7021V",
        "https://drive.google.com/uc?id=1lFploJrtqOvO7ol891iB5ThVtzGvKJRX",
        "https://drive.google.com/uc?id=1fE5Lh1g1te0mqoKauEcSHi7BAphsiRts",
        "https://drive.google.com/uc?id=19U5V-s7z4Yyy_rqatnaKnUO176POfKOH",
        "https://drive.google.com/uc?id=1mtK_Dux36WVIKyik7VA34BdfvzAzhanf",
        "https://drive.google.com/uc?id=1Sc4PARLp1fFSk5WLy5f-7N0kPWATfpxI",
        "https://drive.google.com/uc?id=1UFUyepQMSZ7rVU9xfjaFhtAixkDt_tFD",
        "https://drive.google.com/uc?id=1XDQjt3T4T9yv_kuv-qMDzgp3oWYSd7Ha",
        "https://drive.google.com/uc?id=1l1EYABneZ5DzNXtPeBf55FdvmnzJ6xE4",
        "https://drive.google.com/uc?id=1vVuSjbdag0_9Qf_mdlgSi8tYDSfrb5SL",
        "https://drive.google.com/uc?id=1TGgHOMONepdCksJn_Qw_wh8J_nk-TNZq",
        "https://drive.google.com/uc?id=1_DerFr7Yl0ziVgwbmz2x-wVT2jMfpqD1",
        "https://drive.google.com/uc?id=1PvRBaSWeHp__aF1xUHQrOHRb79kG5GSB",
        "https://drive.google.com/uc?id=1Wq1zjXoVziit-wvFWu_d7hQmoWXpRUvv",
        "https://drive.google.com/uc?id=1M0ISCc5QAF8_W3YhOkeObh5uW5ka1Z5Z",
        "https://drive.google.com/uc?id=1gw3zuLHaO-m6h4tCJSd_1X-3t8ZYGt5Y",
        "https://drive.google.com/uc?id=1qvJd58k-FtW23hW9ST0B5Y1muSNWWAd4",
        "https://drive.google.com/uc?id=1PVW_qto7hr6wn_cE9IOvJY2qyvI0af2f",
        "https://drive.google.com/uc?id=17xIOdPE0iriGwKIohf7xVPp0Coci0_dy",
        "https://drive.google.com/uc?id=1WjyQ1cv-GnyMXcBEZLODAlUGPIUGr95T",
        "https://drive.google.com/uc?id=1nogHaCigFYzLugdxAsZwGsLNF2c-1NzQ",
        "https://drive.google.com/uc?id=10Dli8R_74pdKAgW8v79fIyFzfbgcnVUH",
    ],
    # Just a placeholder for other categories
    "Ladder Operations": [
        "https://drive.google.com/uc?id=1WV2sdHoIU_AI0IdGkQ_KI_ePw81hwfpI",
        # Add more here
    ],
    "Engine Operations": [
        "https://drive.google.com/uc?id=1TQVcN_E4z_GzWEae3tTTt6L8mK9lXku5",
        # Add more here
    ]
}

def load_fdny_pdfs(categories=None):
    selected_links = []
    if categories:
        for cat in categories:
            selected_links.extend(DOC_CATEGORIES.get(cat, []))
    else:
        for links in DOC_CATEGORIES.values():
            selected_links.extend(links)

    temp_dir = tempfile.mkdtemp()
    all_chunks = []

    for url in selected_links:
        try:
            output = os.path.join(temp_dir, url.split("=")[-1] + ".pdf")
            gdown.download(url, output, quiet=True)
            loader = PyMuPDFLoader(output)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"❌ Failed to load: {url} | Error: {e}")

    return all_chunks




