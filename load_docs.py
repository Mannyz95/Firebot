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
    "https://drive.google.com/uc?id=1Qdm_GSRX1b3QTdHawmd-yvPrvTPTEjf0",
    "https://drive.google.com/uc?id=1lIW4RNoWcNCZ3b2AoQgvayzTToi2ScmC",
    "https://drive.google.com/uc?id=19OtGU2KmJNFajkXm2J3DWXrtWeU_hcLL",
    "https://drive.google.com/uc?id=19WnxbQTwG94Khtmb0ljP-GHf9vW4iCIb",
    "https://drive.google.com/uc?id=1hrH6vCRrxGkbgas-x-lono83k7Zm38Nj",
    "https://drive.google.com/uc?id=1AuXutLNahySaD8H__qkZAHgCJVPIh1Km",
    "https://drive.google.com/uc?id=1TuZz-5MHuYkzilJxlmr7CvlpVmglmrnl",
    "https://drive.google.com/uc?id=1vzRcRBEVMP6bzo-iaI9TwbowcIINsUGJ",
    "https://drive.google.com/uc?id=1Y7o9w_0Wn-VxUsJkqsNUxMgk29TcfHPJ",
    "https://drive.google.com/uc?id=1vvV8JUBIPQKFNV1kBmqn_EV9_jpP_uG6",
    "https://drive.google.com/uc?id=1C52gLFLVffc8tXTRMNmNib0N8Mm_kr1C"
,
    ],
    "Engine Operations": [
        "https://drive.google.com/uc?id=1TQVcN_E4z_GzWEae3tTTt6L8mK9lXku5",
    "https://drive.google.com/uc?id=17N7vfJ487bTYo2vWvAWhW4z9JVbp9Xxk",
    "https://drive.google.com/uc?id=1L8uJZCNq80duzQUNuk5EHh7yu89SJ_Xj",
    "https://drive.google.com/uc?id=1ttXDyltMpk_WCy8O3HCAOi8DhYHVLez1",
    "https://drive.google.com/uc?id=19HNDub1pevjLNlB2XIUqTcpvViwcAfXt",
    "https://drive.google.com/uc?id=19yNYqpCRJtbp2W7cGBAPfK4pG3uWrnkL",
    "https://drive.google.com/uc?id=1nvls7Pa5r9CCU_0D4oz2yaUhxJinoX1U",
    "https://drive.google.com/uc?id=1j4iQdVsQRaesF3ko55vANlH3r6YZhoW8",
    "https://drive.google.com/uc?id=1KEYz68jXQHKzNpgThd4Hm3s_dyAZgTXm",
    "https://drive.google.com/uc?id=1jZ_oSkmaVauiYzMwxv2_mQFeadhKlIzI",
    "https://drive.google.com/uc?id=1VQMgo4TL0wd4xfUj-9vYw3XUQ2w2d8Hr",
    "https://drive.google.com/uc?id=1f5XlYj8ynFFiXMDr2pWFINU5NfPgAOmH",
    "https://drive.google.com/uc?id=1z58ih3DTLFD6C6hluNx3_Md_GoIfyQt4",
    "https://drive.google.com/uc?id=1yJuOIe2DD1ho_JwoFSWnaUlVu9xGbEuv",
    "https://drive.google.com/uc?id=1IpTyX71gW6PJvKaFOewqyDuybgGyEEX7",
    "https://drive.google.com/uc?id=1SZUzBEfg14o6_QnQCmHvVRd1iR3RcwaH",
    "https://drive.google.com/uc?id=1l4Ot14JZo2G7kMmSR8HDSqEx4O6QAJXZ",
    "https://drive.google.com/uc?id=1KH2C6Jt47TKbN_LTvYZl0ckW07uaT-8P",
    "https://drive.google.com/uc?id=1_dZa0DQzfkUSuemktMkhn43NdU_-C0CM",
    "https://drive.google.com/uc?id=1jRJxdivhTsF9e6FMpmWyK3E4KYGdiIGV"
    ],
      
}

import os
import gdown
import tempfile
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def normalize_drive_url(url):
    # Handles both "file/d/..." and "open?id=..." formats
    if "id=" in url:
        return url
    elif "file/d/" in url:
        file_id = url.split("/file/d/")[1].split("/")[0]
        return f"https://drive.google.com/uc?id={file_id}"
    else:
        raise ValueError(f"❌ Unrecognized URL format: {url}")

def load_fdny_pdfs(categories=None):
    from load_docs import DOC_CATEGORIES  # Make sure this exists

    selected_links = []
    if categories:
        for cat in categories:
            selected_links.extend(DOC_CATEGORIES.get(cat, []))
    else:
        for links in DOC_CATEGORIES.values():
            selected_links.extend(links)

    temp_dir = tempfile.mkdtemp()
    all_chunks = []

    for raw_url in selected_links:
        try:
            url = normalize_drive_url(raw_url)
            filename = url.split("=")[-1] + ".pdf"
            output = os.path.join(temp_dir, filename)

            print(f"⬇️ Downloading: {url}")
            gdown.download(url, output, quiet=True)

            loader = PyMuPDFLoader(output)
            docs = loader.load()
            if not docs:
                print(f"⚠️ No docs loaded from: {filename}")
                continue

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = splitter.split_documents(docs)
            print(f"✅ Loaded {len(chunks)} chunks from: {filename}")
            all_chunks.extend(chunks)

        except Exception as e:
            print(f"❌ Failed to load: {raw_url}\n   Error: {e}")

    print(f"📦 Total chunks loaded: {len(all_chunks)}")
    return all_chunks





