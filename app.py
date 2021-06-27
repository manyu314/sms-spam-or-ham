from flask import Flask, render_template, request, redirect, url_for
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


app = Flask(__name__)

model = pickle.load(open('sms.pkl', 'rb'))


@app.route('/')
def main():
    return render_template('first.html')


@app.route('/predict', methods=['POST'])
def predict():
    sms = request.form['sms']
    if sms == '':
        return render_template(blank.html)

    # Remove Punctuation
    punct = string.punctuation + '\n'

    def remove_punct(text):
        new_text = ''
        for char in text:
            if char in string.punctuation:
                continue
            else:
                new_text += char
        return new_text

    sms = remove_punct(sms)

    # Stemming
    p_stemmer = PorterStemmer()

    # Function to do the stemming
    def stemming(text):
        text = text.split(' ')  # to seperate the words
        text = [p_stemmer.stem(word) for word in text]

        return text

    sms = stemming(sms)

    # Stopwords

    def remove_stopwords(text):
        text = [
            word for word in text if word not in stopwords.words('english')]

        # converting the list back to text
        # return ' '.join(text)
        return text

    sms = remove_stopwords(sms)

    result = model.predict(sms)

    return render_template('prediction.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
