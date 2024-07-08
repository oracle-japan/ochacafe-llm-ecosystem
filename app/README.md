# app

## how to use

（任意）仮想環境を作成します。

```sh
python3 -m venv .ochat
```

仮想環境を有効化します。

```sh
source .ochat/bin/activate
```

依存ライブラリをダウンロードします。

```sh
pip install -r requirements.txt
```

`.env` ファイルを `.env.example` を確認しながら作成します。

アプリケーションを起動します。

```sh
streamlit run simple-chat-bot.py
```
