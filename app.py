from flask import Flask, render_template, request
from inference import perform_inference, translit_sentence

app = Flask(__name__)



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_text = request.form['input_text']
        output_text = translit_sentence(input_text)  
        return render_template('index.html', input_text=input_text, output_text=output_text)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
