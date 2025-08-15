**Virtual environment & package installation**
```
python3 -m venv /Users/prashantgulati/Documents/dev/python/word_embeddings/.venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Colab version**
https://colab.research.google.com/drive/1J9MgDsVLrBsmNCIkOYmCs1BOqkfolVkR#scrollTo=i6SNxJxFiW9-

**Inspired by**
https://jaketae.github.io/study/word2vec/

**Github**
```
git init
git remote add origin https://github.com/prashant-gulati/wordembeddings
git add .gitignore README.md requirements.txt word_embeddings.py
git commit -m "$(cat <<'EOF'
Initial commit: word embeddings project. Add word_embeddings.py, README, requirements, and .gitignore.
EOF
)"
git push -u origin main 2>&1
```