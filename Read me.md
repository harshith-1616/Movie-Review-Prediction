# 🎬 CineScore: Movie Review Sentiment Predictor

CineScore is a small web app that reads a movie review and tells you whether it is **positive 👍** or **negative 👎**, along with a star rating and a confidence score.

## 🚀 Try it online

**👉 [Open the live app](https://movie-review-prediction-nfs8j3ndepbwyaaicvwcqd.streamlit.app/)**

No installation needed. Just open the link in your browser.
*(If the app has been idle, it may show a "wake up" button. Click it and wait a few seconds.)*

---

## 📖 How to use it

1. Open the app (link above).
2. Type or paste a movie review into the text box.
   Or click one of the sample buttons (🔥 Loved it, 💔 Hated it, 😴 Boring, 😂 Fun ride) to fill in an example.
3. Click **🎬 ANALYZE SENTIMENT**.
4. Read the result card:
   - **THUMBS UP / THUMBS DOWN**: the predicted sentiment
   - **Stars**: how positive the review is (more stars = more positive)
   - **Positive score**: the model's estimated probability that the review is positive
   - **Model confidence**: how sure the model is about its verdict

### Tips for better results
- Write full sentences. Very short or one-word reviews give unreliable results.
- Use English. The model only understands English reviews.
- Reviews with mixed feelings (for example, "great visuals but a boring story") may land close to 50%. That is expected.
- Reviews made up only of numbers, links or symbols are removed during cleaning, and the app will ask you to write something longer.

---

## 🧠 How it works

1. **Cleaning**: your text is lowercased, and links, HTML tags, `[bracketed text]`, punctuation, and words containing numbers are removed.
2. **TF-IDF**: the cleaned text is converted into numbers that show how important each word is.
3. **Logistic Regression**: a trained classifier turns those numbers into a probability of the review being positive.

The trained pipeline is saved in `Review.joblib`.

> ⚠️ This is a machine-learning model, not a human critic. It can make mistakes, especially with sarcasm, slang, and very short reviews.

---

## 💻 Run it on your own computer

### Requirements
- Python 3.9 or newer
- The packages: `streamlit`, `scikit-learn`, `joblib`

### Steps

```bash
# 1. Install the packages
pip install streamlit scikit-learn joblib

# 2. Put these two files in the same folder
#    app.py
#    Review.joblib

# 3. Start the app from that folder
streamlit run app.py
```

Your browser will open at `http://localhost:8501`.

> 💡 Use the same `scikit-learn` version that was used to train the model. A different version can show warnings or fail to load `Review.joblib`.

### Project files

| File | What it does |
|------|--------------|
| `app.py` | The Streamlit web app (interface and prediction code) |
| `Review.joblib` | The saved, trained model (TF-IDF + Logistic Regression) |
| `README.md` | This guide |

---

## ❓ Troubleshooting

| Problem | Fix |
|---------|-----|
| `Review.joblib not found` | Make sure `Review.joblib` is in the same folder as `app.py`. |
| `Can't get attribute 'wordopt'` | Use the provided `app.py`. The model needs the `wordopt` cleaning function defined in it. |
| `'streamlit' is not recognized` | Install it with `pip install streamlit`, then restart your terminal. |
| Fonts look different | The app loads fonts from Google Fonts, so connect to the internet for the full look. |
| "No usable words after cleaning" | Write a longer review with real words. |

---

## 🛠️ Built with

- [Streamlit](https://streamlit.io/) for the web interface
- [scikit-learn](https://scikit-learn.org/) for TF-IDF and Logistic Regression
- [joblib](https://joblib.readthedocs.io/) for loading the saved model
