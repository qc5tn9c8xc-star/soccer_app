import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. タイトルと説明
st.title("サッカーエンブレム判定AI")
st.write("エンブレム画像をアップロードすると、どのリーグか判定します。")

# 2. 学習したモデルを読み込む
# ※さきほど作成したファイル名に合わせてください
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('soccer_emblem_model.h5')

model = load_my_model()

# 3. 画像のアップロード機能
uploaded_file = st.file_uploader("エンブレム画像を選んでください...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 画像を表示
    image = Image.open(uploaded_file)
    st.image(image, caption='アップロードされた画像', use_container_width=True)
    st.write("判定中...")

    # AIが読める形に画像を加工
    img = image.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 4. 判定（予測）
    predictions = model.predict(img_array)
    result = np.argmax(predictions)
    
    # フォルダの順番に合わせてクラス名を設定
    # ※学習した時のフォルダ順（ABC順）にしてください
    classes = ['セリエA', 'ブンデス', 'プレミア', 'ラ・リーガ', 'リーグ・アン']
    
    st.success(f"これは「{classes[result]}」のエンブレムです！")