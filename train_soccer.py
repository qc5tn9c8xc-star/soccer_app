import tensorflow as tf
from tensorflow.keras import layers,models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# 1. 材料の準備
base_dir = "data"     #基本的に参照するフォルダ名
img_size = (150, 150) #画像サイズ。カラーエンブレムは少し細かくみたから150×150くらい
batch_size = 32

#2.画像を読み込む設定(練習用とテスト用)
datagen = ImageDataGenerator(
  rescale=1./255,       #色のデータを０〜１に整える
  validation_split=0.2  #20％をテスト用にする設定
)

#練習用データの読み込み
train_data = datagen.flow_from_directory(
  base_dir,
  target_size = img_size,
  batch_size = batch_size,
  class_mode = "categorical",
  subset = "training"   #80%の練習用
)
#テスト用データの打ち込み
val_data = datagen.flow_from_directory(
  base_dir,
  target_size = img_size,
  batch_size = batch_size,
  class_mode = "categorical",
  subset = "validation"   #20%のテスト用
)

#3.AIの脳の形を作る
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3), activation="relu",input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(5, activation="softmax"))

#4.学習のルールを決める
model.compile(
  optimizer = "adam",
  loss = "categorical_crossentropy",
  metrics = ["accuracy"]
)

#5.学習実行
history = model.fit(
  train_data,
  epochs=10,
  validation_data = val_data
)

#6.できた脳を保存する
# 学習したモデルを保存（名前は何でもOKですが、分かりやすく）
model.save('soccer_emblem_model.h5')
print("モデルの保存が完了しました！")