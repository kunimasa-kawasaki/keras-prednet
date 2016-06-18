# keras-prednet
Deep Predictive Coding NetworksをKerasで実装を試みたもの。

#中身
python train.py  
train.py:学習用  
imageフォルダ:●が移動するサンプルデータです。  

#注意点
ConvLSTMは以下のURLのコードを使わせていただいています。
(https://gist.github.com/henry0312/bc86e166855bc12b18e3bdceb67b3ec1)

PredNetは階層性が特徴ですが、実装は1層のみとなっています。  
制作途中のため正しく学習できるかどうかは不明です。  
BackendはTheanoを使用して確認しています。

#参考文献
  1. Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning(http://arxiv.org/abs/1605.08104)
  3. Unsupervised Learning of Visual Structure using Predictive Generative Networks(http://arxiv.org/abs/1511.06380)
